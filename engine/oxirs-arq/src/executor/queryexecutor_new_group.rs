//! # QueryExecutor - new_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::config::{
    ExecutionContext, ParallelConfig, StreamingResultConfig, ThreadPoolConfig,
};
pub use super::dataset::{
    convert_property_path, ConcreteStoreDataset, Dataset, DatasetPathAdapter, InMemoryDataset,
};
use crate::algebra::{Algebra, Solution};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::queryexecutor_type::QueryExecutor;
use super::types::{ExecutionStrategy, FunctionRegistry};

impl QueryExecutor {
    /// Create new query executor with default configuration
    pub fn new() -> Self {
        let context = ExecutionContext::default();
        let parallel_executor = if context.parallel {
            match crate::parallel::ParallelExecutor::new(context.parallel_config.clone()) {
                Ok(pe) => Some(Arc::new(pe)),
                Err(_) => None,
            }
        } else {
            None
        };
        Self {
            context,
            function_registry: FunctionRegistry::new(),
            parallel_executor,
            result_cache: Arc::new(RwLock::new(HashMap::new())),
            execution_strategy: ExecutionStrategy::default(),
        }
    }
    /// Create executor with custom context
    pub fn with_context(context: ExecutionContext) -> Self {
        let parallel_executor = if context.parallel {
            match crate::parallel::ParallelExecutor::new(context.parallel_config.clone()) {
                Ok(pe) => Some(Arc::new(pe)),
                Err(_) => None,
            }
        } else {
            None
        };
        Self {
            context,
            function_registry: FunctionRegistry::new(),
            parallel_executor,
            result_cache: Arc::new(RwLock::new(HashMap::new())),
            execution_strategy: ExecutionStrategy::default(),
        }
    }
    /// Execute using serial strategy with index-aware optimizations
    pub(super) fn execute_serial(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        match algebra {
            Algebra::Bgp(patterns) => self.execute_bgp_index_aware(patterns, dataset),
            Algebra::Join { left, right } => {
                self.execute_index_optimized_join(left, right, dataset)
            }
            Algebra::Union { left, right } => {
                let left_results = self.execute_serial(left, dataset)?;
                let right_results = self.execute_serial(right, dataset)?;
                Ok(self.union_solutions(left_results, right_results))
            }
            Algebra::Filter { pattern, condition } => {
                let pattern_results = self.execute_serial(pattern, dataset)?;
                self.apply_filter(pattern_results, condition)
            }
            Algebra::Project { pattern, variables } => {
                let pattern_results = self.execute_serial(pattern, dataset)?;
                self.apply_projection(pattern_results, variables)
            }
            Algebra::Distinct { pattern } => {
                let pattern_results = self.execute_serial(pattern, dataset)?;
                Ok(self.apply_distinct(pattern_results))
            }
            Algebra::OrderBy {
                pattern,
                conditions,
            } => {
                let pattern_results = self.execute_serial(pattern, dataset)?;
                Ok(self.apply_order_by(pattern_results, conditions))
            }
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => {
                let pattern_results = self.execute_serial(pattern, dataset)?;
                Ok(self.apply_slice(pattern_results, *offset, *limit))
            }
            Algebra::Group {
                pattern,
                variables,
                aggregates,
            } => {
                let pattern_results = self.execute_serial(pattern, dataset)?;
                self.apply_group_by(pattern_results, variables, aggregates)
            }
            Algebra::LeftJoin {
                left,
                right,
                filter,
            } => {
                let left_results = self.execute_serial(left, dataset)?;
                let right_results = self.execute_serial(right, dataset)?;
                self.apply_left_join(left_results, right_results, filter)
            }
            _ => Ok(Solution::new()),
        }
    }
    /// Execute using streaming strategy
    pub(super) fn execute_streaming(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        use crate::executor::streaming::{StreamingConfig as StreamConfig, StreamingSolution};
        let stream_config = StreamConfig {
            memory_limit: self.context.memory_limit.unwrap_or(1024 * 1024 * 1024),
            temp_dir: None,
            buffer_size: self.context.streaming.buffer_size,
            compress_spills: true,
            spill_strategy: crate::executor::streaming::SpillStrategy::Adaptive,
            adaptive_buffering: true,
            parallel_spilling: true,
            compression_algorithm: crate::executor::streaming::CompressionAlgorithm::Zstd,
        };
        let mut streaming_solution = StreamingSolution::new(stream_config);
        self.execute_algebra_streaming(algebra, dataset, &mut streaming_solution)?;
        streaming_solution.finish();
        let mut result = Solution::new();
        for solution_result in streaming_solution {
            match solution_result {
                Ok(solution) => result.extend(solution),
                Err(e) => return Err(e),
            }
        }
        Ok(result)
    }
    /// Execute algebra expression with streaming support
    pub(super) fn execute_algebra_streaming(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
        streaming_solution: &mut super::streaming::StreamingSolution,
    ) -> Result<()> {
        use crate::executor::streaming::SpillableHashJoin;
        match algebra {
            Algebra::Bgp(patterns) => {
                let bgp_algebra = Algebra::Bgp(patterns.clone());
                let solutions = self.execute_serial(&bgp_algebra, dataset)?;
                for solution in solutions {
                    streaming_solution.add_solution(vec![solution])?;
                }
                Ok(())
            }
            Algebra::Join { left, right } => {
                let left_solutions = self.execute_serial(left, dataset)?;
                let right_solutions = self.execute_serial(right, dataset)?;
                let stream_config = super::streaming::StreamingConfig {
                    memory_limit: self.context.memory_limit.unwrap_or(1024 * 1024 * 1024),
                    temp_dir: None,
                    buffer_size: self.context.streaming.buffer_size,
                    compress_spills: true,
                    spill_strategy: super::streaming::SpillStrategy::Adaptive,
                    adaptive_buffering: true,
                    parallel_spilling: true,
                    compression_algorithm: super::streaming::CompressionAlgorithm::Zstd,
                };
                let mut hash_join = SpillableHashJoin::new(stream_config);
                let join_vars = self.extract_join_variables(left, right);
                let results =
                    hash_join.execute(vec![left_solutions], vec![right_solutions], &join_vars)?;
                for result in results {
                    streaming_solution.add_solution(result)?;
                }
                Ok(())
            }
            Algebra::Union { left, right } => {
                self.execute_algebra_streaming(left, dataset, streaming_solution)?;
                self.execute_algebra_streaming(right, dataset, streaming_solution)?;
                Ok(())
            }
            Algebra::Filter {
                pattern,
                condition: _,
            } => {
                self.execute_algebra_streaming(pattern, dataset, streaming_solution)?;
                Ok(())
            }
            _ => {
                let solution = self.execute_serial(algebra, dataset)?;
                for binding in solution {
                    streaming_solution.add_solution(vec![binding])?;
                }
                Ok(())
            }
        }
    }
    /// Execute algebra in serial mode for update operations
    pub(super) fn execute_serial_algebra(
        &self,
        algebra: &Algebra,
        _context: &mut crate::algebra::EvaluationContext,
    ) -> Result<Solution> {
        match algebra {
            Algebra::Bgp(patterns) => {
                let mut solution = Solution::new();
                if !patterns.is_empty() {
                    let mut binding = crate::algebra::Binding::new();
                    for pattern in patterns {
                        if let crate::algebra::Term::Variable(var) = &pattern.subject {
                            binding.insert(
                                var.clone(),
                                crate::algebra::Term::Iri(
                                    oxirs_core::model::NamedNode::new("http://example.org/subject")
                                        .expect("hardcoded IRI should be valid"),
                                ),
                            );
                        }
                        if let crate::algebra::Term::Variable(var) = &pattern.object {
                            binding.insert(
                                var.clone(),
                                crate::algebra::Term::Iri(
                                    oxirs_core::model::NamedNode::new("http://example.org/object")
                                        .expect("hardcoded IRI should be valid"),
                                ),
                            );
                        }
                    }
                    if !binding.is_empty() {
                        solution.push(binding);
                    }
                }
                Ok(solution)
            }
            _ => Ok(Solution::new()),
        }
    }
    /// Execute BGP with index-aware optimizations
    pub(super) fn execute_bgp_index_aware(
        &self,
        patterns: &[crate::algebra::TriplePattern],
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        if patterns.is_empty() {
            return Ok(Solution::new());
        }
        let stats = crate::optimizer::Statistics::new();
        let index_stats = crate::optimizer::IndexStatistics::default();
        let optimizer = crate::bgp_optimizer::BGPOptimizer::new(&stats, &index_stats);
        let optimized_bgp = optimizer.optimize_bgp(patterns.to_vec())?;
        let mut current_solution =
            self.execute_single_pattern(&optimized_bgp.patterns[0], dataset)?;
        for pattern in optimized_bgp.patterns.iter().skip(1) {
            let pattern_results = self.execute_single_pattern(pattern, dataset)?;
            current_solution = self.join_solutions(current_solution, pattern_results)?;
            if current_solution.is_empty() {
                break;
            }
        }
        Ok(current_solution)
    }
    /// Join two solutions
    pub(super) fn join_solutions(&self, left: Solution, right: Solution) -> Result<Solution> {
        let mut result = Solution::new();
        for left_binding in &left {
            for right_binding in &right {
                let mut is_compatible = true;
                let mut merged = left_binding.clone();
                for (var, term) in right_binding {
                    if let Some(existing_term) = merged.get(var) {
                        if existing_term != term {
                            is_compatible = false;
                            break;
                        }
                    } else {
                        merged.insert(var.clone(), term.clone());
                    }
                }
                if is_compatible {
                    result.push(merged);
                }
            }
        }
        Ok(result)
    }
    /// Apply GROUP BY with aggregation
    pub(super) fn apply_group_by(
        &self,
        solution: Solution,
        variables: &[crate::algebra::GroupCondition],
        aggregates: &[(crate::algebra::Variable, crate::algebra::Aggregate)],
    ) -> Result<Solution> {
        use std::collections::HashMap;
        let mut groups: HashMap<
            Vec<(crate::algebra::Variable, crate::algebra::Term)>,
            Vec<&crate::algebra::Binding>,
        > = HashMap::new();
        for binding in &solution {
            let mut group_key = Vec::new();
            for group_condition in variables {
                if let crate::algebra::Expression::Variable(var) = &group_condition.expr {
                    if let Some(term) = binding.get(var) {
                        group_key.push((var.clone(), term.clone()));
                    }
                }
            }
            groups.entry(group_key).or_default().push(binding);
        }
        if variables.is_empty() && !solution.is_empty() {
            let mut all_bindings = Vec::new();
            for binding in &solution {
                all_bindings.push(binding);
            }
            groups.insert(Vec::new(), all_bindings);
        }
        let mut result = Solution::new();
        for (group_key, group_bindings) in groups {
            let mut group_result = crate::algebra::Binding::new();
            for (var, term) in group_key {
                group_result.insert(var, term);
            }
            for (agg_var, aggregate) in aggregates {
                let agg_value = self.calculate_aggregate(aggregate, &group_bindings)?;
                group_result.insert(agg_var.clone(), agg_value);
            }
            result.push(group_result);
        }
        Ok(result)
    }
    /// Calculate aggregate value
    pub(super) fn calculate_aggregate(
        &self,
        aggregate: &crate::algebra::Aggregate,
        bindings: &[&crate::algebra::Binding],
    ) -> Result<crate::algebra::Term> {
        use std::collections::HashSet;
        match aggregate {
            crate::algebra::Aggregate::Count { distinct, expr } => {
                if let Some(expr) = expr {
                    let mut values = Vec::new();
                    for binding in bindings {
                        if let Ok(value) = self.evaluate_expression(expr, binding) {
                            values.push(value);
                        }
                    }
                    let count = if *distinct {
                        let unique_values: HashSet<_> = values.into_iter().collect();
                        unique_values.len()
                    } else {
                        values.len()
                    };
                    Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                        value: count.to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#integer",
                        )),
                    }))
                } else {
                    let count = bindings.len();
                    Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                        value: count.to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#integer",
                        )),
                    }))
                }
            }
            crate::algebra::Aggregate::Sum { distinct, expr } => {
                let mut values = Vec::new();
                for binding in bindings {
                    if let Ok(value) = self.evaluate_expression(expr, binding) {
                        if let Ok(num) = self.extract_numeric_value(&value) {
                            values.push(num);
                        }
                    }
                }
                if *distinct {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    values.dedup_by(|a, b| a == b);
                }
                let sum: f64 = values.iter().sum();
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: sum.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#decimal",
                    )),
                }))
            }
            crate::algebra::Aggregate::Min { distinct: _, expr } => {
                let mut min_value: Option<f64> = None;
                for binding in bindings {
                    if let Ok(value) = self.evaluate_expression(expr, binding) {
                        if let Ok(num) = self.extract_numeric_value(&value) {
                            min_value = Some(min_value.map_or(num, |min| min.min(num)));
                        }
                    }
                }
                if let Some(min) = min_value {
                    Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                        value: min.to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#decimal",
                        )),
                    }))
                } else {
                    Err(anyhow::anyhow!("No numeric values found for MIN aggregate"))
                }
            }
            crate::algebra::Aggregate::Max { distinct: _, expr } => {
                let mut max_value: Option<f64> = None;
                for binding in bindings {
                    if let Ok(value) = self.evaluate_expression(expr, binding) {
                        if let Ok(num) = self.extract_numeric_value(&value) {
                            max_value = Some(max_value.map_or(num, |max| max.max(num)));
                        }
                    }
                }
                if let Some(max) = max_value {
                    Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                        value: max.to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#decimal",
                        )),
                    }))
                } else {
                    Err(anyhow::anyhow!("No numeric values found for MAX aggregate"))
                }
            }
            crate::algebra::Aggregate::Avg { distinct, expr } => {
                let mut values = Vec::new();
                for binding in bindings {
                    if let Ok(value) = self.evaluate_expression(expr, binding) {
                        if let Ok(num) = self.extract_numeric_value(&value) {
                            values.push(num);
                        }
                    }
                }
                if *distinct {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    values.dedup_by(|a, b| a == b);
                }
                if values.is_empty() {
                    Err(anyhow::anyhow!("No numeric values found for AVG aggregate"))
                } else {
                    let sum: f64 = values.iter().sum();
                    let avg = sum / values.len() as f64;
                    Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                        value: avg.to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#decimal",
                        )),
                    }))
                }
            }
            crate::algebra::Aggregate::Sample { distinct: _, expr } => {
                for binding in bindings {
                    if let Ok(value) = self.evaluate_expression(expr, binding) {
                        return Ok(value);
                    }
                }
                Err(anyhow::anyhow!("No values found for SAMPLE aggregate"))
            }
            crate::algebra::Aggregate::GroupConcat {
                distinct,
                expr,
                separator,
            } => {
                let mut values = Vec::new();
                for binding in bindings {
                    if let Ok(value) = self.evaluate_expression(expr, binding) {
                        let string_value = match value {
                            crate::algebra::Term::Literal(lit) => lit.value,
                            crate::algebra::Term::Iri(iri) => iri.to_string(),
                            crate::algebra::Term::BlankNode(bn) => format!("_{bn}"),
                            _ => value.to_string(),
                        };
                        values.push(string_value);
                    }
                }
                if *distinct {
                    let unique_values: HashSet<_> = values.into_iter().collect();
                    values = unique_values.into_iter().collect();
                    values.sort();
                }
                let sep = separator.as_deref().unwrap_or(" ");
                let concatenated = values.join(sep);
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: concatenated,
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#string",
                    )),
                }))
            }
        }
    }
    /// Apply LEFT JOIN (OPTIONAL)
    pub(super) fn apply_left_join(
        &self,
        left: Solution,
        right: Solution,
        _conditions: &Option<crate::algebra::Expression>,
    ) -> Result<Solution> {
        let mut result = Solution::new();
        for left_binding in &left {
            let mut has_join = false;
            for right_binding in &right {
                let mut is_compatible = true;
                let mut merged = left_binding.clone();
                for (var, term) in right_binding {
                    if let Some(existing_term) = merged.get(var) {
                        if existing_term != term {
                            is_compatible = false;
                            break;
                        }
                    } else {
                        merged.insert(var.clone(), term.clone());
                    }
                }
                if is_compatible {
                    result.push(merged);
                    has_join = true;
                }
            }
            if !has_join {
                result.push(left_binding.clone());
            }
        }
        Ok(result)
    }
    /// Hash join implementation
    pub(super) fn hash_join(&self, build_side: Solution, probe_side: Solution) -> Result<Solution> {
        use std::collections::{HashMap, HashSet};
        let build_vars: HashSet<_> = if let Some(first_binding) = build_side.first() {
            first_binding.keys().collect()
        } else {
            return Ok(Solution::new());
        };
        let probe_vars: HashSet<_> = if let Some(first_binding) = probe_side.first() {
            first_binding.keys().collect()
        } else {
            return Ok(Solution::new());
        };
        let shared_vars: Vec<_> = build_vars.intersection(&probe_vars).cloned().collect();
        let mut hash_table: HashMap<
            Vec<(crate::algebra::Variable, crate::algebra::Term)>,
            Vec<&crate::algebra::Binding>,
        > = HashMap::new();
        for binding in &build_side {
            let key: Vec<_> = shared_vars
                .iter()
                .filter_map(|var| binding.get(var).map(|term| ((*var).clone(), term.clone())))
                .collect();
            hash_table.entry(key).or_default().push(binding);
        }
        let mut result = Solution::new();
        for probe_binding in &probe_side {
            let probe_key: Vec<_> = shared_vars
                .iter()
                .filter_map(|var| {
                    probe_binding
                        .get(var)
                        .map(|term| ((*var).clone(), term.clone()))
                })
                .collect();
            if let Some(matching_bindings) = hash_table.get(&probe_key) {
                for &build_binding in matching_bindings {
                    let mut is_compatible = true;
                    let mut merged = probe_binding.clone();
                    for (var, term) in build_binding {
                        if let Some(existing_term) = merged.get(var) {
                            if existing_term != term {
                                is_compatible = false;
                                break;
                            }
                        } else {
                            merged.insert(var.clone(), term.clone());
                        }
                    }
                    if is_compatible {
                        result.push(merged);
                    }
                }
            }
        }
        Ok(result)
    }
    /// Apply filter to solution
    pub(super) fn apply_filter(
        &self,
        solution: Solution,
        condition: &crate::algebra::Expression,
    ) -> Result<Solution> {
        let mut filtered = Solution::new();
        for binding in solution {
            match self.evaluate_expression(condition, &binding) {
                Ok(crate::algebra::Term::Literal(lit)) => {
                    if self.is_truthy(&lit) {
                        filtered.push(binding);
                    }
                }
                Ok(_) => {
                    filtered.push(binding);
                }
                Err(_) => {}
            }
        }
        Ok(filtered)
    }
}
