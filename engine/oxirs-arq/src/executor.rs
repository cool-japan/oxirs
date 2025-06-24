//! Query Execution Engine
//!
//! This module provides the core query execution engine that evaluates
//! SPARQL algebra expressions and produces result bindings.

use crate::algebra::{Algebra, Binding, Expression, TriplePattern, Term, Variable, Solution, BinaryOperator, UnaryOperator, Aggregate, PropertyPath, PropertyPathPattern};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use rand::Rng;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};

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
            memory_limit: Some(1024 * 1024 * 1024), // 1GB default limit
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
        let results = self.triples.iter()
            .filter(|(s, p, o)| {
                matches_term(&pattern.subject, s) &&
                matches_term(&pattern.predicate, p) &&
                matches_term(&pattern.object, o)
            })
            .cloned()
            .collect();
        Ok(results)
    }
    
    fn contains_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<bool> {
        Ok(self.triples.iter().any(|(s, p, o)| s == subject && p == predicate && o == object))
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
            Some(Arc::new(ParallelExecutor::new(context.parallel_config.clone())))
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
            Some(Arc::new(ParallelExecutor::new(context.parallel_config.clone())))
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
    pub fn execute(&self, algebra: &Algebra, dataset: &dyn Dataset) -> Result<(Solution, ExecutionStats)> {
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
    fn execute_parallel(&self, algebra: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
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
        complexity >= self.context.parallel_threshold
    }
    
    /// Estimate query complexity for parallel execution decision
    fn estimate_complexity(&self, algebra: &Algebra) -> usize {
        match algebra {
            Algebra::Bgp(patterns) => patterns.len() * 100,
            Algebra::Join { left, right } => {
                let left_complexity = self.estimate_complexity(left);
                let right_complexity = self.estimate_complexity(right);
                left_complexity + right_complexity + 500 // Join overhead
            }
            Algebra::Union { left, right } => {
                let left_complexity = self.estimate_complexity(left);
                let right_complexity = self.estimate_complexity(right);
                left_complexity + right_complexity + 200 // Union overhead
            }
            Algebra::Filter { pattern, .. } => {
                self.estimate_complexity(pattern) + 100 // Filter overhead
            }
            Algebra::Service { pattern, .. } => {
                self.estimate_complexity(pattern) + 1000 // Service overhead
            }
            _ => 100, // Default complexity
        }
    }
    
    fn execute_algebra(&self, algebra: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        stats.operations += 1;
        
        match algebra {
            Algebra::Bgp(patterns) => self.execute_bgp(patterns, dataset, stats),
            Algebra::PropertyPath { subject, path, object } => self.execute_property_path(subject, path, object, dataset, stats),
            Algebra::Join { left, right } => self.execute_join(left, right, dataset, stats),
            Algebra::LeftJoin { left, right, filter } => self.execute_left_join(left, right, filter.as_ref(), dataset, stats),
            Algebra::Union { left, right } => self.execute_union(left, right, dataset, stats),
            Algebra::Filter { pattern, condition } => self.execute_filter(pattern, condition, dataset, stats),
            Algebra::Extend { pattern, variable, expr } => self.execute_extend(pattern, variable, expr, dataset, stats),
            Algebra::Minus { left, right } => self.execute_minus(left, right, dataset, stats),
            Algebra::Project { pattern, variables } => self.execute_project(pattern, variables, dataset, stats),
            Algebra::Distinct { pattern } => self.execute_distinct(pattern, dataset, stats),
            Algebra::Slice { pattern, offset, limit } => self.execute_slice(pattern, *offset, *limit, dataset, stats),
            Algebra::OrderBy { pattern, conditions } => self.execute_order_by(pattern, conditions, dataset, stats),
            Algebra::Group { pattern, variables, aggregates } => self.execute_group(pattern, variables, aggregates, dataset, stats),
            Algebra::Having { pattern, condition } => self.execute_having(pattern, condition, dataset, stats),
            Algebra::Service { endpoint, pattern, silent } => self.execute_service(endpoint, pattern, *silent, dataset, stats),
            Algebra::Graph { graph, pattern } => self.execute_graph(graph, pattern, dataset, stats),
            Algebra::Values { variables, bindings } => self.execute_values(variables, bindings, stats),
            Algebra::Table => Ok(vec![HashMap::new()]), // Empty binding
            Algebra::Zero => Ok(vec![]), // No results
            _ => Err(anyhow!("Algebra operation not yet implemented: {:?}", algebra)),
        }
    }
    
    fn execute_bgp(&self, patterns: &[TriplePattern], dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        if patterns.is_empty() {
            return Ok(vec![HashMap::new()]);
        }
        
        // Start with the first pattern
        let mut solution = self.execute_triple_pattern(&patterns[0], dataset, stats)?;
        
        // Join with remaining patterns
        for pattern in patterns.iter().skip(1) {
            let pattern_solution = self.execute_triple_pattern(pattern, dataset, stats)?;
            solution = self.join_solutions(solution, pattern_solution);
            stats.intermediate_results += solution.len();
        }
        
        Ok(solution)
    }
    
    /// Execute property path pattern
    fn execute_property_path(&self, subject: &Term, path: &PropertyPath, object: &Term, 
                           dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        // Property path evaluation using graph traversal
        let mut solution = Vec::new();
        
        // Get all possible subject-object pairs that satisfy the property path
        let paths = self.evaluate_property_path(subject, path, object, dataset, stats)?;
        
        for (subj, obj) in paths {
            let mut binding = HashMap::new();
            
            // Bind variables from subject and object
            if let Term::Variable(var) = subject {
                binding.insert(var.clone(), subj);
            }
            if let Term::Variable(var) = object {
                binding.insert(var.clone(), obj);
            }
            
            solution.push(binding);
        }
        
        Ok(solution)
    }
    
    /// Evaluate property path to find all matching subject-object pairs
    fn evaluate_property_path(&self, subject: &Term, path: &PropertyPath, object: &Term, 
                            dataset: &dyn Dataset, _stats: &mut ExecutionStats) -> Result<Vec<(Term, Term)>> {
        match path {
            PropertyPath::Iri(iri) => {
                // Simple property path - equivalent to triple pattern
                let pattern = TriplePattern::new(subject.clone(), Term::Iri(iri.clone()), object.clone());
                let triples = dataset.find_triples(&pattern)?;
                Ok(triples.into_iter().map(|(s, _p, o)| (s, o)).collect())
            }
            PropertyPath::Variable(var) => {
                // Property variable - find all properties between subject and object
                let pattern = TriplePattern::new(subject.clone(), Term::Variable(var.clone()), object.clone());
                let triples = dataset.find_triples(&pattern)?;
                Ok(triples.into_iter().map(|(s, _p, o)| (s, o)).collect())
            }
            PropertyPath::Inverse(inner_path) => {
                // Inverse path - swap subject and object
                self.evaluate_property_path(object, inner_path, subject, dataset, _stats)
                    .map(|pairs| pairs.into_iter().map(|(o, s)| (s, o)).collect())
            }
            PropertyPath::Sequence(left, right) => {
                // Sequence path - find intermediate nodes
                self.evaluate_sequence_path(subject, left, right, object, dataset, _stats)
            }
            PropertyPath::Alternative(left, right) => {
                // Alternative path - union of both paths
                let mut result = self.evaluate_property_path(subject, left, object, dataset, _stats)?;
                let right_result = self.evaluate_property_path(subject, right, object, dataset, _stats)?;
                result.extend(right_result);
                
                // Remove duplicates
                result.sort();
                result.dedup();
                Ok(result)
            }
            PropertyPath::ZeroOrMore(inner_path) => {
                // Kleene star - transitive closure with reflexivity
                self.evaluate_transitive_closure(subject, inner_path, object, dataset, true, _stats)
            }
            PropertyPath::OneOrMore(inner_path) => {
                // Kleene plus - transitive closure without reflexivity
                self.evaluate_transitive_closure(subject, inner_path, object, dataset, false, _stats)
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                // Optional path - original path union with identity
                let mut result = self.evaluate_property_path(subject, inner_path, object, dataset, _stats)?;
                
                // Add identity relation if subject == object (for variables) or if both are the same constant
                if matches!((subject, object), (Term::Variable(_), Term::Variable(_))) || subject == object {
                    match (subject, object) {
                        (Term::Variable(_), Term::Variable(_)) => {
                            // Add all possible self-loops
                            let all_subjects = dataset.subjects()?;
                            for subj in all_subjects {
                                result.push((subj.clone(), subj));
                            }
                        }
                        _ if subject == object => {
                            result.push((subject.clone(), object.clone()));
                        }
                        _ => {}
                    }
                }
                
                result.sort();
                result.dedup();
                Ok(result)
            }
            PropertyPath::NegatedPropertySet(negated_paths) => {
                // Negated property set - all properties except those in the set
                self.evaluate_negated_property_set(subject, negated_paths, object, dataset, _stats)
            }
        }
    }
    
    /// Evaluate sequence property path (path1/path2)
    fn evaluate_sequence_path(&self, subject: &Term, left: &PropertyPath, right: &PropertyPath, 
                            object: &Term, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Vec<(Term, Term)>> {
        let mut result = Vec::new();
        
        // For sequence paths, we need to find all intermediate nodes
        // Get all possible intermediate terms
        let mut rng = rand::thread_rng();
        let intermediate_var = format!("__intermediate_{}", rng.gen::<u32>());
        let intermediate_term = Term::Variable(intermediate_var);
        
        // First, evaluate left path from subject to intermediate
        let left_pairs = self.evaluate_property_path(subject, left, &intermediate_term, dataset, stats)?;
        
        // For each intermediate result, evaluate right path to object
        for (start, intermediate) in left_pairs {
            let right_pairs = self.evaluate_property_path(&intermediate, right, object, dataset, stats)?;
            for (_, end) in right_pairs {
                result.push((start.clone(), end));
            }
        }
        
        result.sort();
        result.dedup();
        Ok(result)
    }
    
    /// Evaluate transitive closure for * and + operators
    fn evaluate_transitive_closure(&self, subject: &Term, path: &PropertyPath, object: &Term, 
                                 dataset: &dyn Dataset, include_reflexive: bool, 
                                 stats: &mut ExecutionStats) -> Result<Vec<(Term, Term)>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = Vec::new();
        
        // Start with direct connections
        let direct_pairs = self.evaluate_property_path(subject, path, object, dataset, stats)?;
        
        for (s, o) in direct_pairs {
            if !visited.contains(&(s.clone(), o.clone())) {
                visited.insert((s.clone(), o.clone()));
                result.push((s.clone(), o.clone()));
                queue.push((s, o));
            }
        }
        
        // If include_reflexive, add identity relations
        if include_reflexive {
            match (subject, object) {
                (Term::Variable(_), Term::Variable(_)) => {
                    // Add all possible nodes as self-loops
                    let all_nodes = self.get_all_nodes(dataset)?;
                    for node in all_nodes {
                        if !visited.contains(&(node.clone(), node.clone())) {
                            visited.insert((node.clone(), node.clone()));
                            result.push((node.clone(), node));
                        }
                    }
                }
                _ if subject == object => {
                    result.push((subject.clone(), object.clone()));
                }
                _ => {}
            }
        }
        
        // Breadth-first search for transitive closure
        let max_iterations = 1000; // Prevent infinite loops
        let mut iterations = 0;
        
        while !queue.is_empty() && iterations < max_iterations {
            let current_queue = queue.clone();
            queue.clear();
            
            for (current_subject, current_object) in current_queue {
                // Find all nodes reachable from current_object via the path
                let mut rng = rand::thread_rng();
                let next_var = format!("__next_{}", rng.gen::<u32>());
                let next_term = Term::Variable(next_var);
                
                let next_pairs = self.evaluate_property_path(&current_object, path, &next_term, dataset, stats)?;
                
                for (_, next_object) in next_pairs {
                    let new_pair = (current_subject.clone(), next_object.clone());
                    if !visited.contains(&new_pair) {
                        visited.insert(new_pair.clone());
                        result.push(new_pair.clone());
                        queue.push(new_pair);
                    }
                }
            }
            
            iterations += 1;
        }
        
        Ok(result)
    }
    
    /// Evaluate negated property set
    fn evaluate_negated_property_set(&self, subject: &Term, negated_paths: &[PropertyPath], 
                                   object: &Term, dataset: &dyn Dataset, 
                                   stats: &mut ExecutionStats) -> Result<Vec<(Term, Term)>> {
        // Get all predicates in the dataset
        let all_predicates = dataset.predicates()?;
        let mut allowed_predicates = Vec::new();
        
        // Filter out negated predicates
        for predicate in all_predicates {
            let mut is_negated = false;
            
            for negated_path in negated_paths {
                if let PropertyPath::Iri(iri) = negated_path {
                    if predicate == Term::Iri(iri.clone()) {
                        is_negated = true;
                        break;
                    }
                }
            }
            
            if !is_negated {
                allowed_predicates.push(predicate);
            }
        }
        
        // Find triples using allowed predicates
        let mut result = Vec::new();
        for predicate in allowed_predicates {
            let pattern = TriplePattern::new(subject.clone(), predicate, object.clone());
            let triples = dataset.find_triples(&pattern)?;
            result.extend(triples.into_iter().map(|(s, _p, o)| (s, o)));
        }
        
        result.sort();
        result.dedup();
        Ok(result)
    }
    
    /// Get all nodes (subjects and objects) in the dataset
    fn get_all_nodes(&self, dataset: &dyn Dataset) -> Result<Vec<Term>> {
        let mut nodes = HashSet::new();
        
        // Add all subjects
        for subject in dataset.subjects()? {
            nodes.insert(subject);
        }
        
        // Add all objects
        for object in dataset.objects()? {
            nodes.insert(object);
        }
        
        Ok(nodes.into_iter().collect())
    }
    
    fn execute_triple_pattern(&self, pattern: &TriplePattern, dataset: &dyn Dataset, _stats: &mut ExecutionStats) -> Result<Solution> {
        let triples = dataset.find_triples(pattern)?;
        let mut solution = Vec::new();
        
        for (subject, predicate, object) in triples {
            let mut binding = HashMap::new();
            
            // Bind variables from the pattern
            if let Term::Variable(var) = &pattern.subject {
                binding.insert(var.clone(), subject.clone());
            }
            if let Term::Variable(var) = &pattern.predicate {
                binding.insert(var.clone(), predicate.clone());
            }
            if let Term::Variable(var) = &pattern.object {
                binding.insert(var.clone(), object.clone());
            }
            
            solution.push(binding);
        }
        
        Ok(solution)
    }
    
    fn execute_join(&self, left: &Algebra, right: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let left_solution = self.execute_algebra(left, dataset, stats)?;
        let right_solution = self.execute_algebra(right, dataset, stats)?;
        
        Ok(self.join_solutions(left_solution, right_solution))
    }
    
    fn execute_left_join(&self, left: &Algebra, right: &Algebra, filter: Option<&Expression>, 
                        dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let left_solution = self.execute_algebra(left, dataset, stats)?;
        let right_solution = self.execute_algebra(right, dataset, stats)?;
        
        let mut result = Vec::new();
        
        for left_binding in left_solution {
            let mut matched = false;
            
            for right_binding in &right_solution {
                if self.bindings_compatible(&left_binding, right_binding) {
                    let mut combined = left_binding.clone();
                    combined.extend(right_binding.clone());
                    
                    // Apply optional filter
                    if let Some(filter_expr) = filter {
                        if self.evaluate_expression(filter_expr, &combined).unwrap_or(false) {
                            result.push(combined);
                            matched = true;
                        }
                    } else {
                        result.push(combined);
                        matched = true;
                    }
                }
            }
            
            // If no match found, include left binding alone
            if !matched {
                result.push(left_binding);
            }
        }
        
        Ok(result)
    }
    
    fn execute_union(&self, left: &Algebra, right: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let mut left_solution = self.execute_algebra(left, dataset, stats)?;
        let right_solution = self.execute_algebra(right, dataset, stats)?;
        
        left_solution.extend(right_solution);
        Ok(left_solution)
    }
    
    fn execute_filter(&self, pattern: &Algebra, condition: &Expression, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let solution = self.execute_algebra(pattern, dataset, stats)?;
        
        let filtered: Vec<_> = solution.into_iter()
            .filter(|binding| self.evaluate_expression(condition, binding).unwrap_or(false))
            .collect();
        
        Ok(filtered)
    }
    
    fn execute_extend(&self, pattern: &Algebra, variable: &Variable, expr: &Expression, 
                     dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let solution = self.execute_algebra(pattern, dataset, stats)?;
        
        let mut extended = Vec::new();
        for mut binding in solution {
            if let Ok(value) = self.evaluate_expression_term(expr, &binding) {
                binding.insert(variable.clone(), value);
            }
            extended.push(binding);
        }
        
        Ok(extended)
    }
    
    fn execute_minus(&self, left: &Algebra, right: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let left_solution = self.execute_algebra(left, dataset, stats)?;
        let right_solution = self.execute_algebra(right, dataset, stats)?;
        
        let filtered: Vec<_> = left_solution.into_iter()
            .filter(|left_binding| {
                !right_solution.iter().any(|right_binding| 
                    self.bindings_compatible(left_binding, right_binding))
            })
            .collect();
        
        Ok(filtered)
    }
    
    fn execute_project(&self, pattern: &Algebra, variables: &[Variable], dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let solution = self.execute_algebra(pattern, dataset, stats)?;
        
        let projected: Vec<_> = solution.into_iter()
            .map(|binding| {
                variables.iter()
                    .filter_map(|var| binding.get(var).map(|term| (var.clone(), term.clone())))
                    .collect()
            })
            .collect();
        
        Ok(projected)
    }
    
    fn execute_distinct(&self, pattern: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let solution = self.execute_algebra(pattern, dataset, stats)?;
        
        // Simple deduplication without relying on Hash for HashMap
        let mut distinct = Vec::new();
        for binding in solution {
            if !distinct.iter().any(|existing| existing == &binding) {
                distinct.push(binding);
            }
        }
        
        Ok(distinct)
    }
    
    fn execute_slice(&self, pattern: &Algebra, offset: Option<usize>, limit: Option<usize>, 
                    dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let solution = self.execute_algebra(pattern, dataset, stats)?;
        
        let start = offset.unwrap_or(0);
        let end = if let Some(limit) = limit {
            start + limit
        } else {
            solution.len()
        };
        
        Ok(solution.into_iter().skip(start).take(end - start).collect())
    }
    
    fn execute_order_by(&self, pattern: &Algebra, conditions: &[crate::algebra::OrderCondition], 
                       dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let mut solution = self.execute_algebra(pattern, dataset, stats)?;
        
        solution.sort_by(|a, b| {
            for condition in conditions {
                let val_a = self.evaluate_expression_term(&condition.expr, a);
                let val_b = self.evaluate_expression_term(&condition.expr, b);
                
                match (val_a, val_b) {
                    (Ok(term_a), Ok(term_b)) => {
                        let cmp = self.compare_terms(&term_a, &term_b);
                        if cmp != std::cmp::Ordering::Equal {
                            return if condition.ascending { cmp } else { cmp.reverse() };
                        }
                    }
                    _ => continue,
                }
            }
            std::cmp::Ordering::Equal
        });
        
        Ok(solution)
    }
    
    fn join_solutions(&self, left: Solution, right: Solution) -> Solution {
        let mut result = Vec::new();
        
        for left_binding in &left {
            for right_binding in &right {
                if self.bindings_compatible(left_binding, right_binding) {
                    let mut combined = left_binding.clone();
                    combined.extend(right_binding.clone());
                    result.push(combined);
                }
            }
        }
        
        result
    }
    
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
    
    fn evaluate_expression(&self, expr: &Expression, binding: &Binding) -> Result<bool> {
        match expr {
            Expression::Variable(var) => {
                // Variable is bound and not false/zero
                Ok(binding.contains_key(var))
            }
            Expression::Binary { op, left, right } => {
                self.evaluate_binary_expression(op, left, right, binding)
            }
            Expression::Unary { op, expr } => {
                self.evaluate_unary_expression(op, expr, binding)
            }
            Expression::Function { name, args } => {
                self.function_registry.evaluate_boolean_function(name, args, binding)
            }
            _ => Err(anyhow!("Expression evaluation not implemented for: {:?}", expr)),
        }
    }
    
    fn evaluate_expression_term(&self, expr: &Expression, binding: &Binding) -> Result<Term> {
        match expr {
            Expression::Variable(var) => {
                binding.get(var).cloned().ok_or_else(|| anyhow!("Variable {} not bound", var))
            }
            Expression::Literal(lit) => Ok(Term::Literal(lit.clone())),
            Expression::Iri(iri) => Ok(Term::Iri(iri.clone())),
            Expression::Function { name, args } => {
                self.function_registry.evaluate_function(name, args, binding)
            }
            _ => Err(anyhow!("Term evaluation not implemented for: {:?}", expr)),
        }
    }
    
    fn evaluate_binary_expression(&self, op: &BinaryOperator, left: &Expression, right: &Expression, binding: &Binding) -> Result<bool> {
        let left_term = self.evaluate_expression_term(left, binding)?;
        let right_term = self.evaluate_expression_term(right, binding)?;
        
        match op {
            BinaryOperator::Equal => Ok(left_term == right_term),
            BinaryOperator::NotEqual => Ok(left_term != right_term),
            BinaryOperator::Less => Ok(self.compare_terms(&left_term, &right_term) == std::cmp::Ordering::Less),
            BinaryOperator::LessEqual => Ok(self.compare_terms(&left_term, &right_term) != std::cmp::Ordering::Greater),
            BinaryOperator::Greater => Ok(self.compare_terms(&left_term, &right_term) == std::cmp::Ordering::Greater),
            BinaryOperator::GreaterEqual => Ok(self.compare_terms(&left_term, &right_term) != std::cmp::Ordering::Less),
            BinaryOperator::SameTerm => Ok(left_term == right_term),
            _ => Err(anyhow!("Binary operator not implemented: {:?}", op)),
        }
    }
    
    fn evaluate_unary_expression(&self, op: &UnaryOperator, expr: &Expression, binding: &Binding) -> Result<bool> {
        match op {
            UnaryOperator::Not => {
                let result = self.evaluate_expression(expr, binding)?;
                Ok(!result)
            }
            UnaryOperator::IsIri => {
                let term = self.evaluate_expression_term(expr, binding)?;
                Ok(matches!(term, Term::Iri(_)))
            }
            UnaryOperator::IsBlank => {
                let term = self.evaluate_expression_term(expr, binding)?;
                Ok(matches!(term, Term::BlankNode(_)))
            }
            UnaryOperator::IsLiteral => {
                let term = self.evaluate_expression_term(expr, binding)?;
                Ok(matches!(term, Term::Literal(_)))
            }
            _ => Err(anyhow!("Unary operator not implemented: {:?}", op)),
        }
    }
    
    fn compare_terms(&self, left: &Term, right: &Term) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        
        match (left, right) {
            (Term::Literal(l), Term::Literal(r)) => {
                // Try numeric comparison first
                if let (Ok(ln), Ok(rn)) = (l.value.parse::<f64>(), r.value.parse::<f64>()) {
                    ln.partial_cmp(&rn).unwrap_or(Ordering::Equal)
                } else {
                    l.value.cmp(&r.value)
                }
            }
            (Term::Iri(l), Term::Iri(r)) => l.0.cmp(&r.0),
            (Term::Variable(l), Term::Variable(r)) => l.cmp(r),
            (Term::BlankNode(l), Term::BlankNode(r)) => l.cmp(r),
            // Different types: order by type
            (Term::Iri(_), _) => Ordering::Less,
            (_, Term::Iri(_)) => Ordering::Greater,
            (Term::BlankNode(_), _) => Ordering::Less,
            (_, Term::BlankNode(_)) => Ordering::Greater,
            (Term::Literal(_), _) => Ordering::Less,
            (_, Term::Literal(_)) => Ordering::Greater,
        }
    }
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

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
    
    /// Initialize thread pool if not already created
    fn ensure_thread_pool(&self) -> Result<()> {
        let mut pool = self.thread_pool.lock().map_err(|_| anyhow!("Failed to lock thread pool"))?;
        if pool.is_none() {
            *pool = Some(ThreadPool::new(self.config.max_threads)?);
        }
        Ok(())
    }
    
    /// Execute algebra expression in parallel
    pub fn execute(&self, algebra: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        self.ensure_thread_pool()?;
        
        match algebra {
            Algebra::Join { left, right } => self.execute_parallel_join(left, right, dataset, stats),
            Algebra::Union { left, right } => self.execute_parallel_union(left, right, dataset, stats),
            Algebra::Bgp(patterns) => self.execute_parallel_bgp(patterns, dataset, stats),
            _ => {
                // Fallback to sequential execution for unsupported parallel operations
                let executor = QueryExecutor::new();
                executor.execute_algebra(algebra, dataset, stats)
            }
        }
    }
    
    /// Execute parallel join
    fn execute_parallel_join(&self, left: &Algebra, right: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        use std::sync::mpsc;
        
        let (tx, rx) = mpsc::channel();
        let tx1 = tx.clone();
        let tx2 = tx.clone();
        
        let left_algebra = left.clone();
        let right_algebra = right.clone();
        
        // Sequential execution for now to avoid lifetime issues  
        // TODO: Implement proper parallel execution with dataset cloning or shared references
        let executor = QueryExecutor::new();
        let mut left_stats = ExecutionStats::default();
        let left_result = executor.execute_algebra(&left_algebra, dataset, &mut left_stats);
        tx1.send((0, left_result, left_stats)).unwrap();
        
        let mut right_stats = ExecutionStats::default();  
        let right_result = executor.execute_algebra(&right_algebra, dataset, &mut right_stats);
        tx2.send((1, right_result, right_stats)).unwrap();
        
        // Collect results
        let mut left_solution = None;
        let mut right_solution = None;
        
        for _ in 0..2 {
            if let Ok((side, result, side_stats)) = rx.recv() {
                stats.operations += side_stats.operations;
                stats.intermediate_results += side_stats.intermediate_results;
                
                match side {
                    0 => left_solution = Some(result?),
                    1 => right_solution = Some(result?),
                    _ => {}
                }
            }
        }
        
        // Sequential execution completed
        
        // Perform join on results
        if let (Some(left_sol), Some(right_sol)) = (left_solution, right_solution) {
            let executor = QueryExecutor::new();
            Ok(executor.join_solutions(left_sol, right_sol))
        } else {
            Ok(vec![])
        }
    }
    
    /// Execute parallel union
    fn execute_parallel_union(&self, left: &Algebra, right: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        use std::sync::mpsc;
        
        let (tx, rx) = mpsc::channel();
        let tx1 = tx.clone();
        let tx2 = tx.clone();
        
        let left_algebra = left.clone();
        let right_algebra = right.clone();
        
        // Sequential execution for now to avoid lifetime issues
        // TODO: Implement proper parallel execution with dataset cloning or shared references  
        let executor = QueryExecutor::new();
        let mut left_stats = ExecutionStats::default();
        let left_result = executor.execute_algebra(&left_algebra, dataset, &mut left_stats);
        tx1.send((left_result, left_stats)).unwrap();
        
        let mut right_stats = ExecutionStats::default();
        let right_result = executor.execute_algebra(&right_algebra, dataset, &mut right_stats);
        tx2.send((right_result, right_stats)).unwrap();
        
        // Collect results
        let mut combined_solution = Vec::new();
        
        for _ in 0..2 {
            if let Ok((result, side_stats)) = rx.recv() {
                stats.operations += side_stats.operations;
                stats.intermediate_results += side_stats.intermediate_results;
                
                if let Ok(solution) = result {
                    combined_solution.extend(solution);
                }
            }
        }
        
        // Wait for threads to complete
        // Sequential execution completed
        
        Ok(combined_solution)
    }
    
    /// Execute parallel BGP (Basic Graph Pattern)
    fn execute_parallel_bgp(&self, patterns: &[TriplePattern], dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        if patterns.is_empty() {
            return Ok(vec![HashMap::new()]);
        }
        
        // For large BGPs, partition patterns and execute in parallel
        if patterns.len() >= self.config.parallel_threshold / 100 {
            let chunk_size = std::cmp::max(1, patterns.len() / self.config.max_threads);
            let chunks: Vec<_> = patterns.chunks(chunk_size).collect();
            let chunk_count = chunks.len();
            
            use std::sync::mpsc;
            let (tx, rx) = mpsc::channel();
            
            // Sequential execution for now to avoid lifetime issues
            // TODO: Implement proper parallel execution with dataset cloning or shared references
            for (i, chunk) in chunks.into_iter().enumerate() {
                let tx = tx.clone();
                let chunk = chunk.to_vec();
                
                let executor = QueryExecutor::new();
                let mut chunk_stats = ExecutionStats::default();
                let result = executor.execute_bgp(&chunk, dataset, &mut chunk_stats);
                tx.send((i, result, chunk_stats)).unwrap();
            }
            
            // Collect and combine results
            let mut chunk_results = Vec::new();
            for _ in 0..chunk_count {
                if let Ok((_, result, chunk_stats)) = rx.recv() {
                    stats.operations += chunk_stats.operations;
                    stats.intermediate_results += chunk_stats.intermediate_results;
                    
                    if let Ok(solution) = result {
                        chunk_results.push(solution);
                    }
                }
            }
            
            // All chunks completed sequentially
            
            // Join all chunk results
            if chunk_results.is_empty() {
                Ok(vec![])
            } else {
                let executor = QueryExecutor::new();
                let mut result_iter = chunk_results.into_iter();
                let mut result = result_iter.next().unwrap();
                for chunk_result in result_iter {
                    result = executor.join_solutions(result, chunk_result);
                    stats.intermediate_results += result.len();
                }
                Ok(result)
            }
        } else {
            // For small BGPs, use sequential execution
            let executor = QueryExecutor::new();
            executor.execute_bgp(patterns, dataset, stats)
        }
    }
}

impl ThreadPool {
    fn new(size: usize) -> Result<ThreadPool> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(size);
        
        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver))?);
        }
        
        Ok(ThreadPool { workers, sender })
    }
    
    fn execute<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(Message::NewJob(job))
            .map_err(|_| anyhow!("Failed to send job to thread pool"))?;
        Ok(())
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            self.sender.send(Message::Terminate).unwrap();
        }
        
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<std::sync::mpsc::Receiver<Message>>>) -> Result<Worker> {
        let thread = thread::spawn(move || loop {
            let message = receiver.lock().unwrap().recv().unwrap();
            
            match message {
                Message::NewJob(job) => {
                    job();
                }
                Message::Terminate => {
                    break;
                }
            }
        });
        
        Ok(Worker {
            id,
            thread: Some(thread),
        })
    }
}

/// Function registry for custom SPARQL functions
#[derive(Clone)]
pub struct FunctionRegistry {
    functions: HashMap<String, Arc<dyn Fn(&[Expression], &Binding) -> Result<Term> + Send + Sync>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
        };
        
        // Register built-in functions
        registry.register_builtin_functions();
        registry
    }
    
    fn register_builtin_functions(&mut self) {
        // String functions
        self.register_function("str", |args, binding| {
            if args.len() != 1 {
                return Err(anyhow!("str() requires exactly 1 argument"));
            }
            // Implementation would go here
            Err(anyhow!("str() function not yet implemented"))
        });
        
        // Numeric functions
        self.register_function("abs", |args, binding| {
            if args.len() != 1 {
                return Err(anyhow!("abs() requires exactly 1 argument"));
            }
            // Implementation would go here
            Err(anyhow!("abs() function not yet implemented"))
        });
    }
    
    pub fn register_function<F>(&mut self, name: &str, func: F)
    where
        F: Fn(&[Expression], &Binding) -> Result<Term> + Send + Sync + 'static,
    {
        self.functions.insert(name.to_string(), Arc::new(func));
    }
    
    pub fn evaluate_function(&self, name: &str, args: &[Expression], binding: &Binding) -> Result<Term> {
        if let Some(func) = self.functions.get(name) {
            func(args, binding)
        } else {
            Err(anyhow!("Unknown function: {}", name))
        }
    }
    
    pub fn evaluate_boolean_function(&self, name: &str, args: &[Expression], binding: &Binding) -> Result<bool> {
        let result = self.evaluate_function(name, args, binding)?;
        
        // Convert result to boolean
        match result {
            Term::Literal(lit) => {
                match lit.value.as_str() {
                    "true" => Ok(true),
                    "false" => Ok(false),
                    _ => Ok(!lit.value.is_empty()),
                }
            }
            _ => Ok(true), // Non-empty terms are truthy
        }
    }
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryExecutor {
    fn execute_group(&self, pattern: &Algebra, variables: &[crate::algebra::GroupCondition], 
                    aggregates: &[(Variable, crate::algebra::Aggregate)], dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let solution = self.execute_algebra(pattern, dataset, stats)?;
        
        // Group by the specified variables
        let mut groups: HashMap<Vec<Term>, Vec<Binding>> = HashMap::new();
        
        for binding in solution {
            let mut group_key = Vec::new();
            for group_var in variables {
                if let Ok(term) = self.evaluate_expression_term(&group_var.expr, &binding) {
                    group_key.push(term);
                } else {
                    group_key.push(Term::Literal(crate::algebra::Literal {
                        value: "".to_string(),
                        language: None,
                        datatype: None,
                    }));
                }
            }
            groups.entry(group_key).or_insert_with(Vec::new).push(binding);
        }
        
        // Apply aggregates to each group
        let mut result = Vec::new();
        for (group_key, group_bindings) in groups {
            let mut group_binding = HashMap::new();
            
            // Add group variables to binding
            for (i, group_var) in variables.iter().enumerate() {
                if let Some(alias) = &group_var.alias {
                    if let Some(term) = group_key.get(i) {
                        group_binding.insert(alias.clone(), term.clone());
                    }
                }
            }
            
            // Apply aggregates
            for (var, aggregate) in aggregates {
                let aggregate_result = self.apply_aggregate(aggregate, &group_bindings)?;
                group_binding.insert(var.clone(), aggregate_result);
            }
            
            result.push(group_binding);
        }
        
        Ok(result)
    }
    
    fn execute_having(&self, pattern: &Algebra, condition: &Expression, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        let solution = self.execute_algebra(pattern, dataset, stats)?;
        
        let filtered: Vec<_> = solution.into_iter()
            .filter(|binding| self.evaluate_expression(condition, binding).unwrap_or(false))
            .collect();
        
        Ok(filtered)
    }
    
    fn execute_service(&self, _endpoint: &Term, pattern: &Algebra, _silent: bool, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        // For now, execute locally (federation support would be added later)
        self.execute_algebra(pattern, dataset, stats)
    }
    
    fn execute_graph(&self, _graph: &Term, pattern: &Algebra, dataset: &dyn Dataset, stats: &mut ExecutionStats) -> Result<Solution> {
        // For now, ignore graph context and execute pattern
        self.execute_algebra(pattern, dataset, stats)
    }
    
    fn execute_values(&self, variables: &[Variable], bindings: &[Binding], _stats: &mut ExecutionStats) -> Result<Solution> {
        // Simply return the provided bindings, filtered by the specified variables
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
        
        Ok(result)
    }
    
    fn apply_aggregate(&self, aggregate: &crate::algebra::Aggregate, bindings: &[Binding]) -> Result<Term> {
        match aggregate {
            crate::algebra::Aggregate::Count { distinct, expr } => {
                let mut count = 0;
                let mut seen = HashSet::new();
                
                for binding in bindings {
                    if let Some(expr) = expr {
                        if let Ok(term) = self.evaluate_expression_term(expr, binding) {
                            if *distinct {
                                if seen.insert(term) {
                                    count += 1;
                                }
                            } else {
                                count += 1;
                            }
                        }
                    } else {
                        count += 1;
                    }
                }
                
                Ok(Term::Literal(crate::algebra::Literal {
                    value: count.to_string(),
                    language: None,
                    datatype: Some(crate::algebra::Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
                }))
            }
            crate::algebra::Aggregate::Sum { expr, .. } => {
                let mut sum = 0.0;
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_term(expr, binding) {
                        if let Term::Literal(lit) = term {
                            if let Ok(val) = lit.value.parse::<f64>() {
                                sum += val;
                            }
                        }
                    }
                }
                
                Ok(Term::Literal(crate::algebra::Literal {
                    value: sum.to_string(),
                    language: None,
                    datatype: Some(crate::algebra::Iri("http://www.w3.org/2001/XMLSchema#decimal".to_string())),
                }))
            }
            crate::algebra::Aggregate::Min { expr, .. } => {
                let mut min_term: Option<Term> = None;
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_term(expr, binding) {
                        if min_term.is_none() || self.compare_terms(&term, min_term.as_ref().unwrap()) == std::cmp::Ordering::Less {
                            min_term = Some(term);
                        }
                    }
                }
                
                min_term.ok_or_else(|| anyhow!("No values to aggregate"))
            }
            crate::algebra::Aggregate::Max { expr, .. } => {
                let mut max_term: Option<Term> = None;
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_term(expr, binding) {
                        if max_term.is_none() || self.compare_terms(&term, max_term.as_ref().unwrap()) == std::cmp::Ordering::Greater {
                            max_term = Some(term);
                        }
                    }
                }
                
                max_term.ok_or_else(|| anyhow!("No values to aggregate"))
            }
            crate::algebra::Aggregate::Avg { expr, .. } => {
                let mut sum = 0.0;
                let mut count = 0;
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_term(expr, binding) {
                        if let Term::Literal(lit) = term {
                            if let Ok(val) = lit.value.parse::<f64>() {
                                sum += val;
                                count += 1;
                            }
                        }
                    }
                }
                
                if count > 0 {
                    Ok(Term::Literal(crate::algebra::Literal {
                        value: (sum / count as f64).to_string(),
                        language: None,
                        datatype: Some(crate::algebra::Iri("http://www.w3.org/2001/XMLSchema#decimal".to_string())),
                    }))
                } else {
                    Err(anyhow!("No values to aggregate"))
                }
            }
            crate::algebra::Aggregate::Sample { expr, .. } => {
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_term(expr, binding) {
                        return Ok(term);
                    }
                }
                Err(anyhow!("No values to sample"))
            }
            crate::algebra::Aggregate::GroupConcat { expr, separator, .. } => {
                let mut values = Vec::new();
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_term(expr, binding) {
                        if let Term::Literal(lit) = term {
                            values.push(lit.value);
                        } else {
                            values.push(format!("{}", term));
                        }
                    }
                }
                
                let sep = separator.as_deref().unwrap_or(" ");
                let result = values.join(sep);
                
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result,
                    language: None,
                    datatype: None,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{TriplePattern, Term, Iri, Literal};
    
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
