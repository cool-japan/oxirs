//! Parallel query executor core: thread pool, algebra dispatch, and the
//! BGP / join / union / filter / order-by / group-by execution paths.
//!
//! This module defines [`ParallelQueryExecutor`] and the bulk of its query
//! evaluation methods. Property-path, optional/minus, and projection-style
//! operators live in the sibling [`crate::parallel_executor_ops`] module;
//! the scan iterator and work-stealing queue live in
//! [`crate::parallel_executor_queue`].

use crate::algebra::{
    Aggregate, Algebra, Binding, Expression, Literal, Solution, Term as AlgebraTerm, TriplePattern,
    Variable,
};
use crate::executor::stats::ExecutionStats;
use crate::executor::{Dataset, ExecutionContext, ParallelConfig};
use crate::expression::ExpressionEvaluator;
use crate::parallel_types::ParallelStats;
use crate::term::{BindingContext, Term};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use oxirs_core::model::NamedNode;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Parallel query executor with advanced features
pub struct ParallelQueryExecutor {
    pub(crate) config: ParallelConfig,
    pub(crate) stats: Arc<RwLock<ParallelStats>>,
    pub(crate) thread_pool: rayon::ThreadPool,
}

/// Type alias for backward compatibility
pub type ParallelExecutor = ParallelQueryExecutor;

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
    pub(crate) fn execute_parallel_internal(
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
    pub(crate) fn execute_parallel_bgp(
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
                    self.extend_binding_with_pattern(binding, pattern, dataset)
                        .unwrap_or_default()
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
    pub(crate) fn merge_bgp_results(
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
    pub(crate) fn merge_bindings(&self, left: &Binding, right: &Binding) -> Option<Binding> {
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
    pub(crate) fn parallel_distinct(&self, solution: Solution) -> Solution {
        // Create a hash representation of bindings for deduplication
        let seen: DashMap<String, ()> = DashMap::new();

        solution
            .into_par_iter()
            .filter(|binding| {
                // Create a deterministic string representation of the binding
                let mut key_parts: Vec<String> = binding
                    .iter()
                    .map(|(var, term)| format!("{var}={term}"))
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
    pub(crate) fn compare_algebra_terms(
        &self,
        a: &AlgebraTerm,
        b: &AlgebraTerm,
    ) -> std::cmp::Ordering {
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
        let groups_vec: Vec<(Vec<AlgebraTerm>, Vec<Binding>)> = groups.into_iter().collect();

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
}
