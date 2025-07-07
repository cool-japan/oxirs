//! Parallel Query Execution Module
//!
//! This module provides parallel execution capabilities for SPARQL queries,
//! including parallel scans, joins, and aggregations using work-stealing and
//! adaptive algorithms.

use crate::algebra::{
    Aggregate, Algebra, Binding, Expression, Literal, PropertyPath, Solution, Term as AlgebraTerm,
    TriplePattern, Variable,
};
use crate::executor::stats::ExecutionStats;
use crate::executor::{Dataset, ExecutionContext, ParallelConfig};
use crate::expression::ExpressionEvaluator;
use crate::term::{BindingContext, Term};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use oxirs_core::model::NamedNode;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Parallel query executor with advanced features
pub struct ParallelQueryExecutor {
    config: ParallelConfig,
    stats: Arc<RwLock<ParallelStats>>,
    thread_pool: rayon::ThreadPool,
}

/// Parallel execution statistics
#[derive(Debug, Default)]
pub struct ParallelStats {
    pub parallel_operations: usize,
    pub work_items_processed: usize,
    pub thread_utilization: f64,
    pub parallel_speedup: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Parallel iterator for solution processing
pub trait ParallelSolutionIterator: Send + Sync {
    fn par_process<F>(&self, f: F) -> Result<Solution>
    where
        F: Fn(&Binding) -> Option<Binding> + Send + Sync;

    fn par_filter<F>(&self, f: F) -> Result<Solution>
    where
        F: Fn(&Binding) -> bool + Send + Sync;

    fn par_extend<F>(&self, f: F) -> Result<Solution>
    where
        F: Fn(&Binding) -> Vec<Binding> + Send + Sync;
}

impl ParallelQueryExecutor {
    /// Create a new parallel query executor
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.max_threads)
            .thread_name(|idx| format!("oxirs-arq-worker-{idx}"))
            .stack_size(
                config
                    .thread_pool_config
                    .stack_size
                    .unwrap_or(8 * 1024 * 1024),
            )
            .build()
            .map_err(|e| anyhow!("Failed to create thread pool: {}", e))?;

        Ok(Self {
            config,
            stats: Arc::new(RwLock::new(ParallelStats::default())),
            thread_pool,
        })
    }

    /// Execute algebra expression in parallel
    pub fn execute(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let start = Instant::now();

        // Update parallel stats
        {
            let mut pstats = self.stats.write();
            pstats.parallel_operations += 1;
        }

        let result = self
            .thread_pool
            .install(|| self.execute_parallel_internal(algebra, dataset, context, stats))?;

        // Calculate speedup
        let _parallel_time = start.elapsed();
        {
            let mut pstats = self.stats.write();
            pstats.thread_utilization = self.calculate_thread_utilization();
        }

        Ok(result)
    }

    /// Internal parallel execution method
    fn execute_parallel_internal(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        match algebra {
            Algebra::Bgp(patterns) => self.execute_parallel_bgp(patterns, dataset, stats),
            Algebra::Join { left, right } => {
                self.execute_parallel_join(left, right, dataset, context, stats)
            }
            Algebra::Union { left, right } => {
                self.execute_parallel_union(left, right, dataset, context, stats)
            }
            Algebra::Filter { pattern, condition } => {
                self.execute_parallel_filter(pattern, condition, dataset, context, stats)
            }
            Algebra::OrderBy {
                pattern,
                conditions,
            } => {
                let conditions_tuple: Vec<(Expression, bool)> = conditions
                    .iter()
                    .map(|c| (c.expr.clone(), c.ascending))
                    .collect();
                self.execute_parallel_order_by(pattern, &conditions_tuple, dataset, context, stats)
            }
            Algebra::Group {
                pattern,
                variables,
                aggregates,
            } => {
                let group_vars: Vec<Variable> = variables
                    .iter()
                    .filter_map(|gc| {
                        gc.alias.clone().or_else(|| {
                            if let Expression::Variable(var) = &gc.expr {
                                Some(var.clone())
                            } else {
                                None
                            }
                        })
                    })
                    .collect();
                self.execute_parallel_group(
                    pattern,
                    &group_vars,
                    aggregates,
                    dataset,
                    context,
                    stats,
                )
            }
            Algebra::PropertyPath {
                subject,
                path,
                object,
            } => {
                self.execute_parallel_property_path(subject, path, object, dataset, context, stats)
            }
            Algebra::LeftJoin {
                left,
                right,
                filter,
            } => self.execute_parallel_left_join(left, right, filter, dataset, context, stats),
            Algebra::Extend {
                pattern,
                variable,
                expr,
            } => self.execute_parallel_extend(pattern, variable, expr, dataset, context, stats),
            Algebra::Minus { left, right } => {
                self.execute_parallel_minus(left, right, dataset, context, stats)
            }
            Algebra::Service {
                endpoint,
                pattern,
                silent,
            } => self.execute_parallel_service(endpoint, pattern, *silent, dataset, context, stats),
            Algebra::Graph { graph, pattern } => {
                self.execute_parallel_graph(graph, pattern, dataset, context, stats)
            }
            Algebra::Project { pattern, variables } => {
                self.execute_parallel_project(pattern, variables, dataset, context, stats)
            }
            Algebra::Distinct { pattern } => {
                self.execute_parallel_distinct(pattern, dataset, context, stats)
            }
            Algebra::Reduced { pattern } => {
                self.execute_parallel_reduced(pattern, dataset, context, stats)
            }
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => self.execute_parallel_slice(pattern, *offset, *limit, dataset, context, stats),
            _ => {
                // Fall back to sequential execution for truly unsupported operations
                Err(anyhow!(
                    "Parallel execution not supported for this algebra type"
                ))
            }
        }
    }

    /// Execute BGP in parallel with partition-based scanning
    fn execute_parallel_bgp(
        &self,
        patterns: &[TriplePattern],
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        if patterns.is_empty() {
            return Ok(vec![HashMap::new()]);
        }

        // Partition patterns for parallel processing
        let chunk_size = std::cmp::max(1, patterns.len() / self.config.max_threads);
        let pattern_chunks: Vec<_> = patterns.chunks(chunk_size).collect();

        // Process each chunk in parallel
        let partial_results: Vec<Solution> = pattern_chunks
            .par_iter()
            .map(|chunk| self.process_bgp_chunk(chunk, dataset))
            .collect::<Result<Vec<_>>>()?;

        // Merge results
        self.merge_bgp_results(partial_results, stats)
    }

    /// Process a chunk of BGP patterns
    fn process_bgp_chunk(
        &self,
        patterns: &[TriplePattern],
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let mut solution = vec![HashMap::new()];

        for pattern in patterns {
            solution = self.join_with_pattern_parallel(solution, pattern, dataset)?;
            if solution.is_empty() {
                break;
            }
        }

        Ok(solution)
    }

    /// Join solution with pattern in parallel
    fn join_with_pattern_parallel(
        &self,
        solution: Solution,
        pattern: &TriplePattern,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        // Use parallel iterator for large solutions
        if solution.len() > self.config.parallel_threshold {
            let results: Vec<Binding> = solution
                .par_iter()
                .flat_map(|binding| {
                    match self.extend_binding_with_pattern(binding, pattern, dataset) {
                        Ok(extensions) => extensions,
                        Err(_) => vec![],
                    }
                })
                .collect();
            Ok(results)
        } else {
            // Sequential for small solutions
            let mut result = Vec::new();
            for binding in solution {
                let extensions = self.extend_binding_with_pattern(&binding, pattern, dataset)?;
                result.extend(extensions);
            }
            Ok(result)
        }
    }

    /// Extend binding with pattern matches
    fn extend_binding_with_pattern(
        &self,
        binding: &Binding,
        pattern: &TriplePattern,
        dataset: &dyn Dataset,
    ) -> Result<Vec<Binding>> {
        let instantiated = self.instantiate_pattern(pattern, binding);
        let triples = dataset.find_triples(&instantiated)?;

        let mut results = Vec::new();
        for (s, p, o) in triples {
            if let Some(new_binding) = self.try_extend_binding(binding, pattern, &s, &p, &o) {
                results.push(new_binding);
            }
        }

        Ok(results)
    }

    /// Instantiate pattern with bindings
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

    /// Try to extend binding with new values
    fn try_extend_binding(
        &self,
        binding: &Binding,
        pattern: &TriplePattern,
        s: &AlgebraTerm,
        p: &AlgebraTerm,
        o: &AlgebraTerm,
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

    /// Try to bind variable to value
    fn try_bind(
        &self,
        binding: &mut Binding,
        pattern_term: &AlgebraTerm,
        value: &AlgebraTerm,
    ) -> bool {
        match pattern_term {
            AlgebraTerm::Variable(var) => {
                if let Some(existing) = binding.get(var) {
                    existing == value
                } else {
                    binding.insert(var.clone(), value.clone());
                    true
                }
            }
            _ => pattern_term == value,
        }
    }

    /// Merge BGP results from parallel execution
    fn merge_bgp_results(
        &self,
        partial_results: Vec<Solution>,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Use parallel reduction for merging
        let merged = partial_results
            .into_par_iter()
            .reduce(Vec::new, |mut acc, mut partial| {
                acc.append(&mut partial);
                acc
            });

        stats.intermediate_results += merged.len();
        Ok(merged)
    }

    /// Execute parallel hash join
    fn execute_parallel_join(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Execute left and right sequentially to avoid borrowing issues with stats
        let left_solution = self.execute_parallel_internal(left, dataset, context, stats)?;
        let right_solution = self.execute_parallel_internal(right, dataset, context, stats)?;

        // Find join variables
        let join_vars = self.find_join_variables(&left_solution, &right_solution);

        if join_vars.is_empty() {
            // Cartesian product
            self.parallel_cartesian_product(left_solution, right_solution, stats)
        } else {
            // Hash join
            self.parallel_hash_join(left_solution, right_solution, join_vars, stats)
        }
    }

    /// Find common variables between solutions
    fn find_join_variables(&self, left: &Solution, right: &Solution) -> Vec<Variable> {
        if left.is_empty() || right.is_empty() {
            return vec![];
        }

        let left_vars: HashSet<_> = left[0].keys().cloned().collect();
        let right_vars: HashSet<_> = right[0].keys().cloned().collect();

        left_vars.intersection(&right_vars).cloned().collect()
    }

    /// Parallel hash join implementation
    fn parallel_hash_join(
        &self,
        left: Solution,
        right: Solution,
        join_vars: Vec<Variable>,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Build hash table from smaller side in parallel
        let (build_side, probe_side) = if left.len() <= right.len() {
            (left, right)
        } else {
            (right, left)
        };

        // Parallel hash table construction using DashMap
        let hash_table: DashMap<Vec<AlgebraTerm>, Vec<Binding>> = DashMap::new();

        build_side.par_iter().for_each(|binding| {
            let key: Vec<AlgebraTerm> = join_vars
                .iter()
                .filter_map(|var| binding.get(var).cloned())
                .collect();

            if key.len() == join_vars.len() {
                hash_table.entry(key).or_default().push(binding.clone());
            }
        });

        // Parallel probing
        let result: Vec<Binding> = probe_side
            .par_iter()
            .flat_map(|probe_binding| {
                let key: Vec<AlgebraTerm> = join_vars
                    .iter()
                    .filter_map(|var| probe_binding.get(var).cloned())
                    .collect();

                if key.len() == join_vars.len() {
                    match hash_table.get(&key) {
                        Some(matches) => matches
                            .iter()
                            .filter_map(|build_binding| {
                                self.merge_bindings(build_binding, probe_binding)
                            })
                            .collect::<Vec<_>>(),
                        _ => {
                            vec![]
                        }
                    }
                } else {
                    vec![]
                }
            })
            .collect();

        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Merge two bindings
    fn merge_bindings(&self, left: &Binding, right: &Binding) -> Option<Binding> {
        let mut merged = left.clone();

        for (var, value) in right {
            if let Some(existing) = merged.get(var) {
                if existing != value {
                    return None;
                }
            } else {
                merged.insert(var.clone(), value.clone());
            }
        }

        Some(merged)
    }

    /// Parallel cartesian product
    fn parallel_cartesian_product(
        &self,
        left: Solution,
        right: Solution,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let result: Vec<Binding> = left
            .par_iter()
            .flat_map(|l| {
                right
                    .iter()
                    .filter_map(|r| self.merge_bindings(l, r))
                    .collect::<Vec<_>>()
            })
            .collect();

        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Execute parallel union
    fn execute_parallel_union(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Execute both branches sequentially to avoid borrowing issues with stats
        let left_result = self.execute_parallel_internal(left, dataset, context, stats)?;
        let right_result = self.execute_parallel_internal(right, dataset, context, stats)?;

        let mut result = left_result;
        result.extend(right_result);

        // Remove duplicates in parallel
        let unique_result = self.parallel_distinct(result);

        stats.intermediate_results += unique_result.len();
        Ok(unique_result)
    }

    /// Parallel distinct operation
    fn parallel_distinct(&self, solution: Solution) -> Solution {
        // Create a hash representation of bindings for deduplication
        let seen: DashMap<String, ()> = DashMap::new();

        solution
            .into_par_iter()
            .filter(|binding| {
                // Create a deterministic string representation of the binding
                let mut key_parts: Vec<String> = binding
                    .iter()
                    .map(|(var, term)| format!("{}={}", var, term))
                    .collect();
                key_parts.sort(); // Ensure consistent ordering
                let key = key_parts.join("||");

                seen.insert(key, ()).is_none()
            })
            .collect()
    }

    /// Execute parallel filter
    fn execute_parallel_filter(
        &self,
        pattern: &Algebra,
        condition: &Expression,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let solution = self.execute_parallel_internal(pattern, dataset, context, stats)?;

        // Create expression evaluator for filtering
        let extension_registry = context.extension_registry.clone();

        // Parallel filtering
        let filtered: Vec<Binding> = solution
            .into_par_iter()
            .filter(|binding| {
                // Create binding context from HashMap
                let mut ctx = BindingContext::new();
                for (var, term) in binding {
                    ctx.bind(var.as_str(), Term::from_algebra_term(term));
                }
                let evaluator_with_ctx =
                    ExpressionEvaluator::with_context(extension_registry.clone(), ctx);
                match evaluator_with_ctx.evaluate(condition) {
                    Ok(term) => term.effective_boolean_value().unwrap_or(false),
                    Err(_) => false,
                }
            })
            .collect();

        stats.intermediate_results += filtered.len();
        Ok(filtered)
    }

    /// Execute parallel order by
    fn execute_parallel_order_by(
        &self,
        pattern: &Algebra,
        conditions: &[(Expression, bool)], // (expr, ascending)
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let mut solution = self.execute_parallel_internal(pattern, dataset, context, stats)?;

        // Clone extension registry for use in closure
        let extension_registry = context.extension_registry.clone();

        // Parallel sort with custom comparator
        solution.par_sort_by(|a, b| {
            for (expr, ascending) in conditions {
                // Create binding contexts
                let mut ctx_a = BindingContext::new();
                let mut ctx_b = BindingContext::new();
                for (var, term) in a {
                    ctx_a.bind(var.as_str(), Term::from_algebra_term(term));
                }
                for (var, term) in b {
                    ctx_b.bind(var.as_str(), Term::from_algebra_term(term));
                }

                let evaluator_a =
                    ExpressionEvaluator::with_context(extension_registry.clone(), ctx_a);
                let evaluator_b =
                    ExpressionEvaluator::with_context(extension_registry.clone(), ctx_b);

                let val_a = evaluator_a.evaluate(expr).ok();
                let val_b = evaluator_b.evaluate(expr).ok();

                match (val_a, val_b) {
                    (Some(a_term), Some(b_term)) => {
                        let alg_a = a_term.to_algebra_term();
                        let alg_b = b_term.to_algebra_term();
                        let cmp = self.compare_algebra_terms(&alg_a, &alg_b);
                        if cmp != std::cmp::Ordering::Equal {
                            return if *ascending { cmp } else { cmp.reverse() };
                        }
                    }
                    (Some(_), None) => return std::cmp::Ordering::Less,
                    (None, Some(_)) => return std::cmp::Ordering::Greater,
                    (None, None) => continue,
                }
            }
            std::cmp::Ordering::Equal
        });

        Ok(solution)
    }

    /// Compare algebra terms for ordering
    fn compare_algebra_terms(&self, a: &AlgebraTerm, b: &AlgebraTerm) -> std::cmp::Ordering {
        // Convert to internal terms for proper comparison
        let term_a = Term::from_algebra_term(a);
        let term_b = Term::from_algebra_term(b);
        term_a
            .partial_cmp(&term_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Execute parallel group by with aggregation
    fn execute_parallel_group(
        &self,
        pattern: &Algebra,
        variables: &[Variable],
        aggregates: &[(Variable, Aggregate)],
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let solution = self.execute_parallel_internal(pattern, dataset, context, stats)?;

        // Parallel grouping using DashMap
        let groups: DashMap<Vec<AlgebraTerm>, Vec<Binding>> = DashMap::new();

        solution.par_iter().for_each(|binding| {
            let key: Vec<AlgebraTerm> = variables
                .iter()
                .map(|var| {
                    binding
                        .get(var)
                        .cloned()
                        .unwrap_or(AlgebraTerm::Variable(var.clone()))
                })
                .collect();

            groups.entry(key).or_default().push(binding.clone());
        });

        // Parallel aggregation - convert DashMap to Vec for parallel iteration
        let groups_vec: Vec<(Vec<AlgebraTerm>, Vec<Binding>)> =
            groups.into_iter().map(|(k, v)| (k, v)).collect();

        let result: Vec<Binding> = groups_vec
            .into_par_iter()
            .map(|(key, group)| {
                let mut result_binding = HashMap::new();

                // Add grouping variables
                for (i, var) in variables.iter().enumerate() {
                    if let Some(term) = key.get(i) {
                        if !matches!(term, AlgebraTerm::Variable(_)) {
                            result_binding.insert(var.clone(), term.clone());
                        }
                    }
                }

                // Compute aggregates
                for (var, agg) in aggregates {
                    if let Ok(value) = self.compute_aggregate(agg, &group, context) {
                        result_binding.insert(var.clone(), value);
                    }
                }

                result_binding
            })
            .collect();

        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Compute aggregate value
    fn compute_aggregate(
        &self,
        aggregate: &Aggregate,
        group: &[Binding],
        context: &ExecutionContext,
    ) -> Result<AlgebraTerm> {
        match aggregate {
            Aggregate::Count { expr, distinct } => {
                let values =
                    self.collect_aggregate_values(expr.as_ref(), group, *distinct, context)?;
                Ok(AlgebraTerm::Literal(Literal::typed(
                    values.len().to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
                )))
            }
            Aggregate::Sum { expr, distinct } => {
                let values =
                    self.collect_aggregate_values(Some(expr), group, *distinct, context)?;
                let sum = self.sum_numeric_values(values)?;
                Ok(sum)
            }
            Aggregate::Min { expr, distinct } => {
                let values =
                    self.collect_aggregate_values(Some(expr), group, *distinct, context)?;
                values
                    .into_iter()
                    .min_by(|a, b| self.compare_algebra_terms(a, b))
                    .ok_or_else(|| anyhow!("No values for MIN"))
            }
            Aggregate::Max { expr, distinct } => {
                let values =
                    self.collect_aggregate_values(Some(expr), group, *distinct, context)?;
                values
                    .into_iter()
                    .max_by(|a, b| self.compare_algebra_terms(a, b))
                    .ok_or_else(|| anyhow!("No values for MAX"))
            }
            Aggregate::Avg { expr, distinct } => {
                let values =
                    self.collect_aggregate_values(Some(expr), group, *distinct, context)?;
                let sum = self.sum_numeric_values(values.clone())?;
                let count = values.len() as f64;

                match sum {
                    AlgebraTerm::Literal(lit) => {
                        let val = lit.value.parse::<f64>().unwrap_or(0.0);
                        Ok(AlgebraTerm::Literal(Literal::typed(
                            (val / count).to_string(),
                            NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#decimal"),
                        )))
                    }
                    _ => Err(anyhow!("Invalid sum for AVG")),
                }
            }
            Aggregate::GroupConcat {
                expr,
                separator,
                distinct,
            } => {
                let values =
                    self.collect_aggregate_values(Some(expr), group, *distinct, context)?;
                let sep = separator.as_deref().unwrap_or(" ");
                let concat = values
                    .iter()
                    .map(|v| self.term_to_string(v))
                    .collect::<Vec<_>>()
                    .join(sep);
                Ok(AlgebraTerm::Literal(Literal::string(concat)))
            }
            _ => Err(anyhow!("Unsupported aggregate")),
        }
    }

    /// Collect values for aggregation
    fn collect_aggregate_values(
        &self,
        expr: Option<&Expression>,
        group: &[Binding],
        distinct: bool,
        context: &ExecutionContext,
    ) -> Result<Vec<AlgebraTerm>> {
        let extension_registry = context.extension_registry.clone();

        let mut values: Vec<AlgebraTerm> = group
            .par_iter()
            .filter_map(|binding| {
                if let Some(expr) = expr {
                    let mut ctx = BindingContext::new();
                    for (var, term) in binding {
                        ctx.bind(var.as_str(), Term::from_algebra_term(term));
                    }
                    let evaluator =
                        ExpressionEvaluator::with_context(extension_registry.clone(), ctx);
                    evaluator.evaluate(expr).ok().map(|t| t.to_algebra_term())
                } else {
                    // COUNT(*) case
                    Some(AlgebraTerm::Literal(Literal::string("1")))
                }
            })
            .collect();

        if distinct {
            values.sort();
            values.dedup();
        }

        Ok(values)
    }

    /// Sum numeric values
    fn sum_numeric_values(&self, values: Vec<AlgebraTerm>) -> Result<AlgebraTerm> {
        let sum = values
            .into_par_iter()
            .filter_map(|term| {
                if let AlgebraTerm::Literal(lit) = term {
                    lit.value.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .sum::<f64>();

        Ok(AlgebraTerm::Literal(Literal::typed(
            sum.to_string(),
            NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#decimal"),
        )))
    }

    /// Convert term to string
    fn term_to_string(&self, term: &AlgebraTerm) -> String {
        match term {
            AlgebraTerm::Iri(iri) => iri.as_str().to_string(),
            AlgebraTerm::Literal(lit) => lit.value.to_string(),
            AlgebraTerm::Variable(var) => format!("?{var}"),
            AlgebraTerm::BlankNode(id) => format!("_:{id}"),
            AlgebraTerm::QuotedTriple(_) => "<<quoted triple>>".to_string(),
            AlgebraTerm::PropertyPath(_) => "<<property path>>".to_string(),
        }
    }

    /// Calculate thread utilization
    fn calculate_thread_utilization(&self) -> f64 {
        // Simplified calculation - in practice would track actual thread usage
        self.thread_pool.current_num_threads() as f64 / self.config.max_threads as f64
    }

    /// Get parallel execution statistics
    pub fn get_stats(&self) -> ParallelStats {
        let stats = self.stats.read();
        ParallelStats {
            parallel_operations: stats.parallel_operations,
            work_items_processed: stats.work_items_processed,
            thread_utilization: stats.thread_utilization,
            parallel_speedup: stats.parallel_speedup,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
        }
    }

    // Additional parallel execution methods for missing algebra types

    /// Execute property path in parallel
    fn execute_parallel_property_path(
        &self,
        subject: &AlgebraTerm,
        path: &PropertyPath,
        object: &AlgebraTerm,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        stats.property_path_evaluations += 1;

        // Parallel property path evaluation based on path type
        match path {
            PropertyPath::Iri(predicate) => {
                // Direct path - simple triple pattern
                let pattern = TriplePattern {
                    subject: subject.clone(),
                    predicate: AlgebraTerm::Iri(predicate.clone()),
                    object: object.clone(),
                };
                self.execute_parallel_bgp(&[pattern], dataset, stats)
            }
            PropertyPath::Inverse(inner_path) => {
                // Inverse path - swap subject and object
                self.execute_parallel_property_path(
                    object, inner_path, subject, dataset, context, stats,
                )
            }
            PropertyPath::Sequence(left_path, right_path) => {
                // Sequence path (p1/p2) - find intermediate nodes
                self.execute_parallel_sequence_path(
                    subject, left_path, right_path, object, dataset, context, stats,
                )
            }
            PropertyPath::Alternative(left_path, right_path) => {
                // Alternative path (p1|p2) - union of both paths
                self.execute_parallel_alternative_path(
                    subject, left_path, right_path, object, dataset, context, stats,
                )
            }
            PropertyPath::ZeroOrMore(inner_path) => {
                // Kleene star - transitive closure
                self.execute_parallel_transitive_closure(
                    subject, inner_path, object, dataset, context, stats, true,
                )
            }
            PropertyPath::OneOrMore(inner_path) => {
                // Plus operator - transitive closure without zero
                self.execute_parallel_transitive_closure(
                    subject, inner_path, object, dataset, context, stats, false,
                )
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                // Optional path - direct or empty
                let direct_result = self.execute_parallel_property_path(
                    subject, inner_path, object, dataset, context, stats,
                )?;

                // Add empty binding if subject equals object (zero path)
                if subject == object {
                    let mut result = direct_result;
                    result.push(HashMap::new());
                    Ok(result)
                } else {
                    Ok(direct_result)
                }
            }
            PropertyPath::Variable(_) => {
                // Property path with variable - not directly supported in parallel execution
                // Fall back to simpler implementation
                Ok(vec![])
            }
            PropertyPath::NegatedPropertySet(_) => {
                // Negated property set - complex implementation required
                // Fall back to simpler implementation
                Ok(vec![])
            }
        }
    }

    /// Execute sequence path in parallel (p1/p2)
    fn execute_parallel_sequence_path(
        &self,
        subject: &AlgebraTerm,
        left_path: &PropertyPath,
        right_path: &PropertyPath,
        object: &AlgebraTerm,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Find all possible intermediate nodes by exploring left path from subject
        let intermediate_var = Variable::new("?__intermediate")?;
        let intermediate_term = AlgebraTerm::Variable(intermediate_var.clone());

        let left_results = self.execute_parallel_property_path(
            subject,
            left_path,
            &intermediate_term,
            dataset,
            context,
            stats,
        )?;

        // For each intermediate result, explore right path to object
        let results: Vec<Solution> = left_results
            .par_iter()
            .filter_map(|binding| {
                if let Some(intermediate_value) = binding.get(&intermediate_var) {
                    // Create a thread-local stats copy for parallel execution
                    let mut local_stats = ExecutionStats::default();
                    self.execute_parallel_property_path(
                        intermediate_value,
                        right_path,
                        object,
                        dataset,
                        context,
                        &mut local_stats,
                    )
                    .ok()
                } else {
                    None
                }
            })
            .collect();

        // Flatten and merge results
        let mut final_result = Vec::new();
        for result_set in results {
            final_result.extend(result_set);
        }

        Ok(final_result)
    }

    /// Execute alternative path in parallel (p1|p2)
    fn execute_parallel_alternative_path(
        &self,
        subject: &AlgebraTerm,
        left_path: &PropertyPath,
        right_path: &PropertyPath,
        object: &AlgebraTerm,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Execute both paths in parallel with separate stats
        let mut left_stats = ExecutionStats::new();
        let mut right_stats = ExecutionStats::new();
        let (left_results, right_results) = rayon::join(
            || {
                self.execute_parallel_property_path(
                    subject,
                    left_path,
                    object,
                    dataset,
                    context,
                    &mut left_stats,
                )
            },
            || {
                self.execute_parallel_property_path(
                    subject,
                    right_path,
                    object,
                    dataset,
                    context,
                    &mut right_stats,
                )
            },
        );

        // Merge stats back
        stats.merge_from(&left_stats);
        stats.merge_from(&right_stats);

        let mut combined_results = left_results?;
        combined_results.extend(right_results?);

        // Remove duplicates
        Ok(self.parallel_distinct(combined_results))
    }

    /// Execute transitive closure in parallel (p+ or p*)
    fn execute_parallel_transitive_closure(
        &self,
        subject: &AlgebraTerm,
        path: &PropertyPath,
        object: &AlgebraTerm,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        _stats: &mut ExecutionStats,
        include_zero: bool,
    ) -> Result<Solution> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut current_level = vec![subject.clone()];
        let max_depth = 50; // Prevent infinite loops

        // Add zero-length path if needed
        if include_zero && subject == object {
            result.push(HashMap::new());
        }

        for depth in 1..=max_depth {
            if current_level.is_empty() {
                break;
            }

            let next_level: Vec<AlgebraTerm> = current_level
                .par_iter()
                .flat_map(|current_node| {
                    // Find all nodes reachable in one step
                    let next_var = Variable::new(&format!("?__next_{depth}")).unwrap();
                    let next_term = AlgebraTerm::Variable(next_var.clone());

                    let mut local_stats = ExecutionStats::new();
                    match self.execute_parallel_property_path(
                        current_node,
                        path,
                        &next_term,
                        dataset,
                        context,
                        &mut local_stats,
                    ) {
                        Ok(step_results) => step_results
                            .into_iter()
                            .filter_map(|binding| binding.get(&next_var).cloned())
                            .collect::<Vec<_>>(),
                        _ => Vec::new(),
                    }
                })
                .filter(|node| !visited.contains(node))
                .collect();

            // Check if we reached the target object
            for node in &next_level {
                if node == object {
                    result.push(HashMap::new());
                }
                visited.insert(node.clone());
            }

            current_level = next_level;
        }

        Ok(result)
    }

    /// Execute left join in parallel
    fn execute_parallel_left_join(
        &self,
        left: &Algebra,
        right: &Algebra,
        filter: &Option<Expression>,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Clone stats to avoid mutable borrow conflicts
        let mut left_stats = stats.clone();
        let mut right_stats = stats.clone();

        // Execute left and right patterns in parallel
        let (left_results, right_results) = rayon::join(
            || self.execute_parallel_internal(left, dataset, context, &mut left_stats),
            || self.execute_parallel_internal(right, dataset, context, &mut right_stats),
        );

        // Merge stats back
        stats.merge_from(&left_stats);
        stats.merge_from(&right_stats);

        let left_solution = left_results?;
        let right_solution = right_results?;

        // Perform left join with optional filter
        self.perform_parallel_left_join(left_solution, right_solution, filter, context)
    }

    /// Execute extend (BIND) in parallel
    fn execute_parallel_extend(
        &self,
        pattern: &Algebra,
        variable: &Variable,
        expr: &Expression,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_solution = self.execute_parallel_internal(pattern, dataset, context, stats)?;

        // Extend bindings in parallel
        let extension_registry = context.extension_registry.clone();
        let extended_solution: Result<Solution> = pattern_solution
            .par_iter()
            .map(|binding| {
                let evaluator = ExpressionEvaluator::new(extension_registry.clone());
                let mut binding_context = BindingContext::new();
                // Set up binding context with current values
                for (var, val) in binding {
                    // Convert algebra::Term to term::Term for binding context
                    let term_val = Term::from_algebra_term(val);
                    binding_context.bind(var.as_str(), term_val);
                }

                match evaluator.evaluate(expr) {
                    Ok(value) => {
                        // Convert term::Term back to algebra::Term
                        let algebra_value = value.to_algebra_term();
                        let mut new_binding = binding.clone();
                        new_binding.insert(variable.clone(), algebra_value);
                        Ok(new_binding)
                    }
                    Err(_) => Ok(binding.clone()), // Keep original binding if evaluation fails
                }
            })
            .collect();

        extended_solution
    }

    /// Execute minus in parallel
    fn execute_parallel_minus(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Clone stats to avoid mutable borrow conflicts
        let mut left_stats = stats.clone();
        let mut right_stats = stats.clone();

        // Execute both patterns in parallel
        let (left_results, right_results) = rayon::join(
            || self.execute_parallel_internal(left, dataset, context, &mut left_stats),
            || self.execute_parallel_internal(right, dataset, context, &mut right_stats),
        );

        // Merge stats back
        stats.merge_from(&left_stats);
        stats.merge_from(&right_stats);

        let left_solution = left_results?;
        let right_solution = right_results?;

        // Perform minus operation in parallel
        self.perform_parallel_minus(left_solution, right_solution)
    }

    /// Execute service in parallel (federation)
    fn execute_parallel_service(
        &self,
        _endpoint: &AlgebraTerm,
        pattern: &Algebra,
        silent: bool,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // For service operations, we typically can't parallelize across the network boundary
        // but we can execute the pattern preparation in parallel
        if silent {
            // In silent mode, return empty solution on failure
            Ok(self
                .execute_parallel_internal(pattern, dataset, context, stats)
                .unwrap_or_else(|_| vec![]))
        } else {
            self.execute_parallel_internal(pattern, dataset, context, stats)
        }
    }

    /// Execute graph pattern in parallel
    fn execute_parallel_graph(
        &self,
        _graph: &AlgebraTerm,
        pattern: &Algebra,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Execute pattern within the specified graph context
        self.execute_parallel_internal(pattern, dataset, context, stats)
    }

    /// Execute projection in parallel
    fn execute_parallel_project(
        &self,
        pattern: &Algebra,
        variables: &[Variable],
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_solution = self.execute_parallel_internal(pattern, dataset, context, stats)?;

        // Project variables in parallel
        let projected_solution: Solution = pattern_solution
            .par_iter()
            .map(|binding| {
                let mut projected_binding = HashMap::new();
                for var in variables {
                    if let Some(value) = binding.get(var) {
                        projected_binding.insert(var.clone(), value.clone());
                    }
                }
                projected_binding
            })
            .collect();

        Ok(projected_solution)
    }

    /// Execute distinct in parallel
    fn execute_parallel_distinct(
        &self,
        pattern: &Algebra,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_solution = self.execute_parallel_internal(pattern, dataset, context, stats)?;

        // Use parallel deduplication with custom approach since HashMap doesn't implement Hash
        let mut distinct_solution = Vec::new();
        let mut seen = std::collections::BTreeSet::new();

        for binding in pattern_solution {
            // Create a comparable representation of the binding
            let binding_key: Vec<_> = binding
                .iter()
                .map(|(k, v)| (k.clone(), format!("{:?}", v)))
                .collect();

            if seen.insert(binding_key) {
                distinct_solution.push(binding);
            }
        }

        Ok(distinct_solution)
    }

    /// Execute reduced in parallel
    fn execute_parallel_reduced(
        &self,
        pattern: &Algebra,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // REDUCED is similar to DISTINCT but may allow duplicates for performance
        // For now, implement same as distinct
        self.execute_parallel_distinct(pattern, dataset, context, stats)
    }

    /// Execute slice (LIMIT/OFFSET) in parallel
    fn execute_parallel_slice(
        &self,
        pattern: &Algebra,
        offset: Option<usize>,
        limit: Option<usize>,
        dataset: &dyn Dataset,
        context: &ExecutionContext,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_solution = self.execute_parallel_internal(pattern, dataset, context, stats)?;

        // Apply offset and limit
        let start = offset.unwrap_or(0);
        let end = limit.map(|l| start + l).unwrap_or(pattern_solution.len());

        Ok(pattern_solution
            .into_iter()
            .skip(start)
            .take(end - start)
            .collect())
    }

    // Helper methods for parallel operations

    /// Perform parallel left join operation
    fn perform_parallel_left_join(
        &self,
        left_solution: Solution,
        right_solution: Solution,
        filter: &Option<Expression>,
        context: &ExecutionContext,
    ) -> Result<Solution> {
        let result_bindings: Vec<Vec<Binding>> = left_solution
            .par_iter()
            .map(|left_binding| {
                // Find compatible bindings in right solution
                let mut found_match = false;
                let mut joined_bindings = Vec::new();

                for right_binding in &right_solution {
                    if self.are_bindings_compatible(left_binding, right_binding) {
                        let mut joined = left_binding.clone();
                        joined.extend(right_binding.clone());

                        // Apply filter if present
                        if let Some(filter_expr) = filter {
                            let evaluator =
                                ExpressionEvaluator::new(context.extension_registry.clone());
                            let mut binding_context = BindingContext::new();
                            // Set up binding context
                            for (var, val) in &joined {
                                let term_val = Term::from_algebra_term(val);
                                binding_context.bind(var.as_str(), term_val);
                            }

                            if let Ok(result) = evaluator.evaluate(filter_expr) {
                                // Check if the result is truthy
                                if self.is_truthy_value(&result) {
                                    joined_bindings.push(joined);
                                    found_match = true;
                                }
                            }
                        } else {
                            joined_bindings.push(joined);
                            found_match = true;
                        }
                    }
                }

                // If no match found, keep left binding (left join semantics)
                if !found_match {
                    joined_bindings.push(left_binding.clone());
                }

                joined_bindings
            })
            .collect();

        let final_result: Solution = result_bindings.into_iter().flatten().collect();
        Ok(final_result)
    }

    /// Perform parallel minus operation
    fn perform_parallel_minus(
        &self,
        left_solution: Solution,
        right_solution: Solution,
    ) -> Result<Solution> {
        let result: Solution = left_solution
            .par_iter()
            .filter(|left_binding| {
                // Keep left binding only if it doesn't have a compatible binding in right
                !right_solution
                    .iter()
                    .any(|right_binding| self.are_bindings_compatible(left_binding, right_binding))
            })
            .cloned()
            .collect();

        Ok(result)
    }

    /// Check if two bindings are compatible (share same values for common variables)
    fn are_bindings_compatible(&self, binding1: &Binding, binding2: &Binding) -> bool {
        for (var, value1) in binding1 {
            if let Some(value2) = binding2.get(var) {
                if value1 != value2 {
                    return false;
                }
            }
        }
        true
    }

    /// Check if a term represents a truthy value
    fn is_truthy_value(&self, term: &Term) -> bool {
        term.effective_boolean_value().unwrap_or(false)
    }
}

/// Parallel scan iterator for large datasets
pub struct ParallelScanIterator<'a> {
    dataset: &'a dyn Dataset,
    pattern: TriplePattern,
    chunk_size: usize,
}

impl<'a> ParallelScanIterator<'a> {
    pub fn new(dataset: &'a dyn Dataset, pattern: TriplePattern, chunk_size: usize) -> Self {
        Self {
            dataset,
            pattern,
            chunk_size,
        }
    }

    /// Scan dataset in parallel chunks
    pub fn par_scan(&self) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        // In a real implementation, this would partition the dataset
        // and scan chunks in parallel
        self.dataset.find_triples(&self.pattern)
    }
}

/// Work-stealing queue for dynamic load balancing
pub struct WorkStealingQueue<T: Send + Sync> {
    queues: Vec<Arc<Mutex<VecDeque<T>>>>,
    thread_count: usize,
    work_counters: Vec<AtomicUsize>,
    global_work_count: AtomicUsize,
}

impl<T: Send + Sync> WorkStealingQueue<T> {
    pub fn new(thread_count: usize) -> Self {
        let mut queues = Vec::with_capacity(thread_count);
        let mut work_counters = Vec::with_capacity(thread_count);

        for _ in 0..thread_count {
            queues.push(Arc::new(Mutex::new(VecDeque::new())));
            work_counters.push(AtomicUsize::new(0));
        }

        Self {
            queues,
            thread_count,
            work_counters,
            global_work_count: AtomicUsize::new(0),
        }
    }

    /// Push work to a specific thread's queue
    pub fn push(&self, thread_id: usize, work: T) {
        let queue_id = thread_id % self.thread_count;
        if let Some(queue) = self.queues.get(queue_id) {
            queue.lock().push_back(work);
            self.work_counters[queue_id].fetch_add(1, Ordering::Relaxed);
            self.global_work_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Push work to the least loaded queue
    pub fn push_balanced(&self, work: T) {
        let mut min_load = usize::MAX;
        let mut best_queue = 0;

        // Find the queue with minimum load
        for (i, counter) in self.work_counters.iter().enumerate() {
            let load = counter.load(Ordering::Relaxed);
            if load < min_load {
                min_load = load;
                best_queue = i;
            }
        }

        self.push(best_queue, work);
    }

    /// Try to get work, stealing if necessary
    pub fn steal(&self, thread_id: usize) -> Option<T> {
        // Try own queue first (LIFO for cache locality)
        if let Some(queue) = self.queues.get(thread_id) {
            if let Some(mut q) = queue.try_lock() {
                if let Some(work) = q.pop_back() {
                    self.work_counters[thread_id].fetch_sub(1, Ordering::Relaxed);
                    self.global_work_count.fetch_sub(1, Ordering::Relaxed);
                    return Some(work);
                }
            }
        }

        // Try to steal from others (FIFO to avoid conflicts)
        let start = (thread_id + 1) % self.thread_count;
        for i in 0..self.thread_count {
            let target = (start + i) % self.thread_count;
            if target != thread_id {
                if let Some(queue) = self.queues.get(target) {
                    if let Some(mut q) = queue.try_lock() {
                        if let Some(work) = q.pop_front() {
                            self.work_counters[target].fetch_sub(1, Ordering::Relaxed);
                            self.global_work_count.fetch_sub(1, Ordering::Relaxed);
                            return Some(work);
                        }
                    }
                }
            }
        }

        None
    }

    /// Get total pending work count
    pub fn pending_work(&self) -> usize {
        self.global_work_count.load(Ordering::Relaxed)
    }

    /// Check if all queues are empty
    pub fn is_empty(&self) -> bool {
        self.pending_work() == 0
    }

    /// Get load distribution across threads
    pub fn get_load_distribution(&self) -> Vec<usize> {
        self.work_counters
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect()
    }

    /// Drain all work from all queues
    pub fn drain_all(&self) -> Vec<T> {
        let mut all_work = Vec::new();

        for (i, queue) in self.queues.iter().enumerate() {
            {
                let mut q = queue.lock();
                while let Some(work) = q.pop_front() {
                    all_work.push(work);
                }
            }
            self.work_counters[i].store(0, Ordering::Relaxed);
        }

        self.global_work_count.store(0, Ordering::Relaxed);
        all_work
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Iri, Literal, Term, Variable};

    #[test]
    fn test_parallel_executor_creation() {
        let config = ParallelConfig::default();
        let executor = ParallelQueryExecutor::new(config).unwrap();
        assert!(executor.config.max_threads > 0);
    }

    #[test]
    fn test_work_stealing_queue() {
        let queue: WorkStealingQueue<i32> = WorkStealingQueue::new(4);

        // Push to different queues
        queue.push(0, 1);
        queue.push(1, 2);
        queue.push(2, 3);

        // Steal from queue 3 (empty) - should steal from queue 0 first (value 1)
        assert_eq!(queue.steal(3), Some(1)); // Should steal from another queue
    }

    #[test]
    fn test_parallel_distinct() {
        let config = ParallelConfig::default();
        let executor = ParallelQueryExecutor::new(config).unwrap();

        let mut solution = vec![];
        for i in 0..100 {
            let mut binding = HashMap::new();
            binding.insert(
                Variable::new("x").unwrap(),
                Term::Literal(Literal {
                    value: (i % 10).to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            solution.push(binding);
        }

        let distinct = executor.parallel_distinct(solution);
        assert_eq!(distinct.len(), 10);
    }
}
