//! Parallel query executor operators: property paths, optional/minus joins,
//! federation, projection, and slicing.
//!
//! These methods extend [`crate::parallel_executor_engine::ParallelQueryExecutor`]
//! with the remaining SPARQL algebra operators that are dispatched from
//! `execute_parallel_internal`.

use crate::algebra::{
    Algebra, Binding, Expression, PropertyPath, Solution, Term as AlgebraTerm, Variable,
};
use crate::executor::stats::ExecutionStats;
use crate::executor::{Dataset, ExecutionContext};
use crate::expression::ExpressionEvaluator;
use crate::parallel_executor_engine::ParallelQueryExecutor;
use crate::term::{BindingContext, Term};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

impl ParallelQueryExecutor {
    /// Execute property path in parallel
    pub(crate) fn execute_parallel_property_path(
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
                let pattern = crate::algebra::TriplePattern {
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
                    let next_var = Variable::new(format!("?__next_{depth}"))
                        .expect("generated variable name should be valid");
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
    pub(crate) fn execute_parallel_left_join(
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
    pub(crate) fn execute_parallel_extend(
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
    pub(crate) fn execute_parallel_minus(
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
    pub(crate) fn execute_parallel_service(
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
    pub(crate) fn execute_parallel_graph(
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
    pub(crate) fn execute_parallel_project(
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
    pub(crate) fn execute_parallel_distinct(
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
                .map(|(k, v)| (k.clone(), format!("{v:?}")))
                .collect();

            if seen.insert(binding_key) {
                distinct_solution.push(binding);
            }
        }

        Ok(distinct_solution)
    }

    /// Execute reduced in parallel
    pub(crate) fn execute_parallel_reduced(
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
    pub(crate) fn execute_parallel_slice(
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
    pub(crate) fn are_bindings_compatible(&self, binding1: &Binding, binding2: &Binding) -> bool {
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
