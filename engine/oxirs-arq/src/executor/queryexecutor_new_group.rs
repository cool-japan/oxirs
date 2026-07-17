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
use crate::algebra::{Aggregate, Algebra, Expression, Solution, Variable};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::queryexecutor_type::QueryExecutor;
use super::types::{ExecutionStrategy, FunctionRegistry, UnknownFunctionError};

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
            adaptive_stats: Arc::new(crate::optimizer::adaptive::AdaptiveStatsStore::new(1024)),
            sla_gate: None,
            execution_budget: None,
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
            adaptive_stats: Arc::new(crate::optimizer::adaptive::AdaptiveStatsStore::new(1024)),
            sla_gate: None,
            execution_budget: None,
        }
    }
    /// Attach an SLA gate to this executor.
    ///
    /// When set, queries can be routed through
    /// [`QueryExecutor::execute_for_tenant`] which performs admission control
    /// before delegating to [`QueryExecutor::execute`].
    pub fn with_sla_gate(mut self, gate: crate::sla_integration::ArqSlaGate) -> Self {
        self.sla_gate = Some(gate);
        self
    }
    /// Whether an SLA gate has been attached.
    pub fn has_sla_gate(&self) -> bool {
        self.sla_gate.is_some()
    }
    /// Borrow the attached SLA gate, if any.
    pub fn sla_gate(&self) -> Option<&crate::sla_integration::ArqSlaGate> {
        self.sla_gate.as_ref()
    }
    /// Attach a runtime resource budget to this executor.
    ///
    /// When set, [`QueryExecutor::execute`] will:
    ///
    /// 1. Call [`crate::query_governor::ExecutionBudget::check_time`] at
    ///    entry, returning early if the wall-time limit is already breached.
    /// 2. Call [`crate::query_governor::ExecutionBudget::record_result_row`]
    ///    once for every binding in the produced solution.
    ///
    /// Triple-scan recording via
    /// [`crate::query_governor::ExecutionBudget::record_triple_scan`] is
    /// provided by the `ExecutionBudget` API but is not yet threaded into
    /// `execute_single_pattern` — callers that need per-triple enforcement
    /// should call it directly in their BGP iterator loops.
    pub fn with_budget(
        mut self,
        budget: std::sync::Arc<crate::query_governor::ExecutionBudget>,
    ) -> Self {
        self.execution_budget = Some(budget);
        self
    }
    /// Whether a runtime budget has been attached.
    pub fn has_budget(&self) -> bool {
        self.execution_budget.is_some()
    }
    /// Borrow the attached budget, if any.
    pub fn budget(&self) -> Option<&std::sync::Arc<crate::query_governor::ExecutionBudget>> {
        self.execution_budget.as_ref()
    }
    /// Execute a query on behalf of `tenant_id`, gated by the attached
    /// [`crate::sla_integration::ArqSlaGate`].
    ///
    /// On admission, the algebra is enqueued in the priority dispatcher (for
    /// observability), then dispatched immediately via [`Self::execute`].
    /// On rejection, [`crate::sla_integration::ArqSlaError::SlaExceeded`] is
    /// returned without touching the dataset.
    ///
    /// Returns an `Err(anyhow::Error)` when no gate is attached or when the
    /// underlying execution fails; the error chain carries the SLA root cause
    /// when applicable.
    pub fn execute_for_tenant(
        &mut self,
        tenant_id: &str,
        algebra: &crate::algebra::Algebra,
        dataset: &dyn super::dataset::Dataset,
    ) -> Result<(crate::algebra::Solution, super::stats::ExecutionStats)> {
        let gate = self
            .sla_gate
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no SLA gate attached to QueryExecutor"))?;
        let admitted = gate
            .admit(tenant_id, ())
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        // Stick the admitted entry into the priority dispatcher for observability,
        // then immediately dequeue (single-thread mode).
        gate.enqueue(&admitted, format!("query for {tenant_id}"));
        let _ = gate.next_dispatch();
        self.execute(algebra, dataset)
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
                self.apply_filter_with_dataset(pattern_results, condition, dataset)
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
            Algebra::Minus { left, right } => self.execute_minus(left, right, dataset),
            Algebra::Reduced { pattern } => {
                // REDUCED is a hint that the engine may (but is not required to) remove
                // duplicate solutions.  The simplest correct implementation is a passthrough
                // that returns all solutions unchanged.
                self.execute_serial(pattern, dataset)
            }
            Algebra::Extend {
                pattern,
                variable,
                expr,
            } => self.execute_extend(pattern, variable, expr, dataset),
            Algebra::Values {
                variables,
                bindings,
            } => self.execute_values(variables, bindings),
            Algebra::PropertyPath {
                subject,
                path,
                object,
            } => self.execute_property_path(subject, path, object, dataset),
            Algebra::Graph { graph, pattern } => self.execute_graph(graph, pattern, dataset),
            Algebra::Having { pattern, condition } => {
                self.execute_having(pattern, condition, dataset)
            }
            Algebra::Service {
                endpoint,
                pattern,
                silent,
            } => crate::service_federation::execute_service_clause(endpoint, pattern, *silent),
            // Unit table (join identity): one solution with no bindings. This is
            // the left operand produced by the parser for leading BIND/OPTIONAL,
            // so it MUST yield a single empty row, not zero rows.
            Algebra::Table => {
                let solution: Solution = vec![crate::algebra::Binding::new()];
                Ok(solution)
            }
            // Zero / Empty are genuinely empty result sets.
            Algebra::Zero | Algebra::Empty => Ok(Solution::new()),
        }
    }
    /// Evaluate a `GRAPH` pattern with real named-graph scoping.
    ///
    /// * `GRAPH <iri> { P }` restricts evaluation of `P` to the named graph
    ///   `<iri>` by wrapping the dataset in a [`super::dataset::GraphScopedDataset`]
    ///   view, so the inner pattern code reads only that graph.
    /// * `GRAPH ?g { P }` enumerates the dataset's named graphs
    ///   ([`Dataset::named_graphs`]), evaluates `P` scoped to each, and extends
    ///   every produced row with `?g` bound to that graph. If the inner pattern
    ///   itself already binds `?g`, rows whose binding disagrees with the
    ///   enumerated graph are dropped (self-consistency); the join-time
    ///   compatibility check handles the case where an *outer* operator has
    ///   already fixed `?g`.
    pub(super) fn execute_graph(
        &self,
        graph: &crate::algebra::Term,
        pattern: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        use super::dataset::{GraphScopedDataset, GraphSelector};
        use crate::algebra::Term;
        match graph {
            Term::Iri(iri) => {
                let scoped = GraphScopedDataset::new(dataset, GraphSelector::Named(iri.clone()));
                self.execute_serial(pattern, &scoped)
            }
            Term::Variable(var) => {
                let graphs = dataset.named_graphs()?;
                let mut combined = Solution::new();
                for g in graphs {
                    let iri = match &g {
                        Term::Iri(n) => n.clone(),
                        // Named graphs are always IRIs; ignore anything else.
                        _ => continue,
                    };
                    let scoped = GraphScopedDataset::new(dataset, GraphSelector::Named(iri));
                    let rows = self.execute_serial(pattern, &scoped)?;
                    for mut binding in rows {
                        match binding.get(var) {
                            Some(existing) if existing != &g => continue,
                            Some(_) => combined.push(binding),
                            None => {
                                binding.insert(var.clone(), g.clone());
                                combined.push(binding);
                            }
                        }
                    }
                }
                Ok(combined)
            }
            other => Err(anyhow::anyhow!(
                "GRAPH label must be an IRI or a variable, got: {other}"
            )),
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
    /// Execute algebra in serial mode for update operations.
    ///
    /// This helper has no access to a dataset, so it cannot evaluate a BGP (or
    /// any pattern that reads the store) against real data. Rather than
    /// fabricate bindings — which previously mapped every subject/object
    /// variable to hardcoded `http://example.org/...` constants and silently
    /// corrupted DELETE/INSERT WHERE — it fails loudly. Callers that need to
    /// evaluate WHERE clauses must run [`QueryExecutor::execute`] against a real
    /// [`Dataset`] (see `update::UpdateExecutor::evaluate_pattern`).
    pub(super) fn execute_serial_algebra(
        &self,
        algebra: &Algebra,
        _context: &mut crate::algebra::EvaluationContext,
    ) -> Result<Solution> {
        match algebra {
            // An empty/unit table is representable without a dataset.
            Algebra::Table => {
                let solution: Solution = vec![crate::algebra::Binding::new()];
                Ok(solution)
            }
            Algebra::Zero | Algebra::Empty => Ok(Solution::new()),
            other => Err(anyhow::anyhow!(
                "execute_serial_algebra cannot evaluate {:?} without a dataset; \
                 use QueryExecutor::execute against a real Dataset instead",
                std::mem::discriminant(other)
            )),
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
        // When there are no GROUP BY variables, ensure we always have exactly one group
        // (even for empty solutions, so COUNT(*) returns 0 rather than no rows)
        if variables.is_empty() {
            groups.entry(Vec::new()).or_default();
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
                Err(err) => {
                    // An unknown function is a whole-query fault: fail the entire
                    // filter loudly rather than silently shrinking the result set
                    // (no-silent-empty contract). Every OTHER error class is a
                    // per-row evaluation error (unbound variable, type error, ...)
                    // which SPARQL 1.1 §17.3 treats as excluding that one row, so
                    // it is swallowed and the row is dropped.
                    if err.downcast_ref::<UnknownFunctionError>().is_some() {
                        return Err(err);
                    }
                }
            }
        }
        Ok(filtered)
    }

    /// Apply filter with dataset access (for EXISTS/NOT EXISTS evaluation)
    pub(super) fn apply_filter_with_dataset(
        &self,
        solution: Solution,
        condition: &crate::algebra::Expression,
        dataset: &dyn super::dataset::Dataset,
    ) -> Result<Solution> {
        use super::queryexecutor_queries::EXISTS_DATASET;
        // Set the dataset pointer in thread-local storage so that EXISTS/NOT EXISTS
        // subquery evaluation can access it
        // Store the dataset as a (data_ptr, vtable_ptr) usize pair in thread-local storage
        // to avoid lifetime issues with storing references in thread-locals.
        let dataset_ptr: *const dyn super::dataset::Dataset = dataset;
        let (data_addr, vtable_addr): (usize, usize) = unsafe { std::mem::transmute(dataset_ptr) };
        EXISTS_DATASET.with(|cell| {
            *cell.borrow_mut() = Some((data_addr, vtable_addr));
        });

        let result = self.apply_filter(solution, condition);

        // Clear the thread-local after filter evaluation
        EXISTS_DATASET.with(|cell| {
            *cell.borrow_mut() = None;
        });

        result
    }

    /// Evaluate a `HAVING` clause over a grouped solution.
    ///
    /// `HAVING` is a post-aggregation `FILTER`, but its condition may itself
    /// contain aggregate function calls (`HAVING (COUNT(?s) > 1)`,
    /// `HAVING (SUM(?a * ?b) >= 10)`). A plain post-group filter cannot evaluate
    /// those, because the per-group source bindings have already been collapsed
    /// into one row per group. This method therefore hoists every aggregate call
    /// found in the condition into a synthetic group aggregate, evaluates it per
    /// group *alongside* the declared aggregates, rewrites the condition to
    /// reference the synthetic alias, applies the filter, and finally strips the
    /// synthetic columns so they do not leak into downstream operators.
    ///
    /// Any caller that builds `Group` + `Having` straight from parsed SPARQL thus
    /// gets correct semantics without pre-rewriting the algebra.
    ///
    /// A condition with no aggregate calls (a plain-variable `HAVING` over
    /// grouping keys / declared aggregates) takes the original fast path: execute
    /// the pattern and filter it directly, behavior-identical to before.
    pub(super) fn execute_having(
        &self,
        pattern: &Algebra,
        condition: &Expression,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        // Detect and hoist aggregate calls in the condition.
        let mut hoisted: Vec<(Variable, Aggregate)> = Vec::new();
        let mut counter = 0usize;
        let rewritten = rewrite_having_aggregates(condition, &mut hoisted, &mut counter)?;

        if hoisted.is_empty() {
            // Plain-variable HAVING: a straight post-aggregation FILTER.
            let grouped = self.execute_serial(pattern, dataset)?;
            return self.apply_filter_with_dataset(grouped, &rewritten, dataset);
        }

        let synthetic_vars: Vec<Variable> = hoisted.iter().map(|(var, _)| var.clone()).collect();

        // Evaluate the hoisted aggregates per group. When the HAVING pattern is a
        // Group (the normal case), append them to its aggregate list so they are
        // computed over the same groups as the declared aggregates. When there is
        // no explicit Group (an aggregate HAVING with neither GROUP BY nor a
        // projected aggregate), the aggregate implies the single implicit group.
        let augmented = match pattern {
            Algebra::Group {
                pattern: inner,
                variables,
                aggregates,
            } => {
                let mut aggregates = aggregates.clone();
                aggregates.extend(hoisted);
                Algebra::Group {
                    pattern: inner.clone(),
                    variables: variables.clone(),
                    aggregates,
                }
            }
            other => Algebra::Group {
                pattern: Box::new(other.clone()),
                variables: Vec::new(),
                aggregates: hoisted,
            },
        };

        let grouped = self.execute_serial(&augmented, dataset)?;
        let filtered = self.apply_filter_with_dataset(grouped, &rewritten, dataset)?;

        // Strip the synthetic aggregate columns so downstream operators
        // (projection, ORDER BY, ...) never observe them.
        let cleaned = filtered
            .into_iter()
            .map(|mut binding| {
                for var in &synthetic_vars {
                    binding.remove(var);
                }
                binding
            })
            .collect();
        Ok(cleaned)
    }
}

/// Rewrite a `HAVING` condition so every aggregate function call is replaced by a
/// reference to a freshly-allocated synthetic group aggregate.
///
/// Aggregate calls (`COUNT(?s)`, `SUM(?a * ?b)`, ...) parse as
/// [`Expression::Function`]; each is hoisted into `aggregates` under a synthetic
/// `__having_agg_N` alias and replaced in the condition with
/// [`Expression::Variable`], so the resulting condition references only
/// grouped / aggregate variables that the post-group filter can evaluate.
/// Non-aggregate function calls and composite expressions are walked recursively;
/// leaves are returned unchanged.
fn rewrite_having_aggregates(
    expr: &Expression,
    aggregates: &mut Vec<(Variable, Aggregate)>,
    counter: &mut usize,
) -> Result<Expression> {
    match expr {
        Expression::Function { name, args } => {
            if let Some(aggregate) = function_to_aggregate(name, args)? {
                let alias = Variable::new(format!("__having_agg_{counter}"))
                    .map_err(|e| anyhow::anyhow!("invalid synthetic HAVING variable: {e}"))?;
                *counter += 1;
                aggregates.push((alias.clone(), aggregate));
                Ok(Expression::Variable(alias))
            } else {
                let rewritten_args = args
                    .iter()
                    .map(|a| rewrite_having_aggregates(a, aggregates, counter))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Expression::Function {
                    name: name.clone(),
                    args: rewritten_args,
                })
            }
        }
        Expression::Binary { op, left, right } => Ok(Expression::Binary {
            op: op.clone(),
            left: Box::new(rewrite_having_aggregates(left, aggregates, counter)?),
            right: Box::new(rewrite_having_aggregates(right, aggregates, counter)?),
        }),
        Expression::Unary { op, operand } => Ok(Expression::Unary {
            op: op.clone(),
            operand: Box::new(rewrite_having_aggregates(operand, aggregates, counter)?),
        }),
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => Ok(Expression::Conditional {
            condition: Box::new(rewrite_having_aggregates(condition, aggregates, counter)?),
            then_expr: Box::new(rewrite_having_aggregates(then_expr, aggregates, counter)?),
            else_expr: Box::new(rewrite_having_aggregates(else_expr, aggregates, counter)?),
        }),
        other => Ok(other.clone()),
    }
}

/// Recognize a SPARQL aggregate function call by name and build the corresponding
/// [`Aggregate`], or return `Ok(None)` for a non-aggregate function.
///
/// Name recognition and arity validation are delegated to the shared helpers
/// [`crate::algebra::aggregate_function_name`] and
/// [`crate::algebra::check_aggregate_arity`], so this executor-side check and the
/// parse-time `HAVING` validator agree on both which calls are aggregates and the
/// exact wrong-arity error texts. This remains as defense-in-depth: callers that
/// build the algebra directly (bypassing the parser) still get a loud, correct
/// rejection of wrong-arity aggregate calls.
fn function_to_aggregate(name: &str, args: &[Expression]) -> Result<Option<Aggregate>> {
    let Some(canonical) = crate::algebra::aggregate_function_name(name) else {
        return Ok(None);
    };
    crate::algebra::check_aggregate_arity(name, args.len()).map_err(|msg| anyhow::anyhow!(msg))?;
    // Arity is validated above: `COUNT` has 0 or 1 argument, every other
    // aggregate exactly one, so the indexing below cannot go out of bounds.
    let aggregate = match canonical {
        "COUNT" => Aggregate::Count {
            distinct: false,
            expr: args.first().cloned(),
        },
        "SUM" => Aggregate::Sum {
            distinct: false,
            expr: args[0].clone(),
        },
        "MIN" => Aggregate::Min {
            distinct: false,
            expr: args[0].clone(),
        },
        "MAX" => Aggregate::Max {
            distinct: false,
            expr: args[0].clone(),
        },
        "AVG" => Aggregate::Avg {
            distinct: false,
            expr: args[0].clone(),
        },
        "SAMPLE" => Aggregate::Sample {
            distinct: false,
            expr: args[0].clone(),
        },
        "GROUP_CONCAT" => Aggregate::GroupConcat {
            distinct: false,
            expr: args[0].clone(),
            separator: None,
        },
        // `aggregate_function_name` only returns the seven names handled above.
        _ => return Ok(None),
    };
    Ok(Some(aggregate))
}

#[cfg(test)]
mod serial_executor_tests {
    use crate::algebra::{
        Algebra, BinaryOperator, Expression, GroupCondition, Literal, Term, TriplePattern, Variable,
    };
    use crate::executor::dataset::InMemoryDataset;
    use crate::executor::QueryExecutor;
    use oxirs_core::model::NamedNode;

    fn v(name: &str) -> Term {
        Term::Variable(Variable::new_unchecked(name))
    }

    fn iri(s: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(s))
    }

    fn ev(name: &str) -> Expression {
        Expression::Variable(Variable::new_unchecked(name))
    }

    fn int_expr(n: i64) -> Expression {
        Expression::Literal(Literal {
            value: n.to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        })
    }

    fn sample_dataset() -> InMemoryDataset {
        InMemoryDataset::from_triples(vec![
            (iri("http://ex/s1"), iri("http://ex/p"), iri("http://ex/o1")),
            (iri("http://ex/s2"), iri("http://ex/p"), iri("http://ex/o2")),
        ])
    }

    fn bgp_s_p_o() -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: v("s"),
            predicate: iri("http://ex/p"),
            object: v("o"),
        }])
    }

    /// A dataset with one default-graph triple and two triples in the named
    /// graph `<http://ex/g>`.
    fn graph_scoped_dataset() -> InMemoryDataset {
        let g = NamedNode::new_unchecked("http://ex/g");
        let mut ds = InMemoryDataset::new();
        ds.add_triple(iri("http://ex/sd"), iri("http://ex/p"), iri("http://ex/od"));
        ds.add_triple_in_graph(
            g.clone(),
            iri("http://ex/s1"),
            iri("http://ex/p"),
            iri("http://ex/o1"),
        );
        ds.add_triple_in_graph(
            g,
            iri("http://ex/s2"),
            iri("http://ex/p"),
            iri("http://ex/o2"),
        );
        ds
    }

    #[test]
    fn execute_serial_graph_iri_scopes_to_that_graph() {
        let exec = QueryExecutor::new();
        let ds = graph_scoped_dataset();
        let algebra = Algebra::Graph {
            graph: iri("http://ex/g"),
            pattern: Box::new(bgp_s_p_o()),
        };
        let sol = exec.execute_serial(&algebra, &ds).expect("graph serial");
        assert_eq!(
            sol.len(),
            2,
            "GRAPH <g> must return only the two triples in <g>, not the default graph"
        );
    }

    #[test]
    fn execute_serial_plain_bgp_reads_default_graph_only() {
        let exec = QueryExecutor::new();
        let ds = graph_scoped_dataset();
        // Plain BGP (no GRAPH) must see only the single default-graph triple,
        // NOT the union with the named graph.
        let sol = exec.execute_serial(&bgp_s_p_o(), &ds).expect("bgp serial");
        assert_eq!(
            sol.len(),
            1,
            "plain BGP must read the default graph only (regression for union bug)"
        );
    }

    #[test]
    fn execute_serial_graph_variable_enumerates_and_binds() {
        let exec = QueryExecutor::new();
        let ds = graph_scoped_dataset();
        let var_g = Variable::new_unchecked("g");
        let algebra = Algebra::Graph {
            graph: Term::Variable(var_g.clone()),
            pattern: Box::new(bgp_s_p_o()),
        };
        let sol = exec
            .execute_serial(&algebra, &ds)
            .expect("graph var serial");
        assert_eq!(
            sol.len(),
            2,
            "GRAPH ?g must enumerate the one named graph and bind both its rows"
        );
        for binding in &sol {
            assert_eq!(
                binding.get(&var_g),
                Some(&iri("http://ex/g")),
                "?g must be bound to the enumerated named graph"
            );
        }
    }

    #[test]
    fn execute_serial_having_filters_grouped_solution() {
        let exec = QueryExecutor::new();
        let ds = sample_dataset();
        let condition = Expression::Binary {
            op: BinaryOperator::Equal,
            left: Box::new(Expression::Variable(Variable::new_unchecked("o"))),
            right: Box::new(Expression::Iri(NamedNode::new_unchecked("http://ex/o1"))),
        };
        let algebra = Algebra::Having {
            pattern: Box::new(bgp_s_p_o()),
            condition,
        };
        let sol = exec.execute_serial(&algebra, &ds).expect("having serial");
        assert_eq!(
            sol.len(),
            1,
            "HAVING must filter, not silently drop to empty"
        );
    }

    /// `?s <dept> ?d` with three subjects in two departments (eng x2, sales x1).
    fn dept_dataset() -> InMemoryDataset {
        InMemoryDataset::from_triples(vec![
            (
                iri("http://ex/s1"),
                iri("http://ex/dept"),
                iri("http://ex/eng"),
            ),
            (
                iri("http://ex/s2"),
                iri("http://ex/dept"),
                iri("http://ex/eng"),
            ),
            (
                iri("http://ex/s3"),
                iri("http://ex/dept"),
                iri("http://ex/sales"),
            ),
        ])
    }

    fn bgp_s_dept_d() -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: v("s"),
            predicate: iri("http://ex/dept"),
            object: v("d"),
        }])
    }

    #[test]
    fn having_count_aggregate_filters_groups_natively() {
        // GROUP BY ?d HAVING (COUNT(?s) > 1): the engine must hoist COUNT(?s) into
        // a per-group aggregate and keep only the eng group (size 2), dropping
        // sales (size 1) — with no caller-side rewrite of the algebra.
        let exec = QueryExecutor::new();
        let ds = dept_dataset();
        let group = Algebra::Group {
            pattern: Box::new(bgp_s_dept_d()),
            variables: vec![GroupCondition {
                expr: ev("d"),
                alias: None,
            }],
            aggregates: Vec::new(),
        };
        let condition = Expression::Binary {
            op: BinaryOperator::Greater,
            left: Box::new(Expression::Function {
                name: "COUNT".to_string(),
                args: vec![ev("s")],
            }),
            right: Box::new(int_expr(1)),
        };
        let algebra = Algebra::Having {
            pattern: Box::new(group),
            condition,
        };
        let sol = exec.execute_serial(&algebra, &ds).expect("having count");
        assert_eq!(
            sol.len(),
            1,
            "HAVING(COUNT(?s) > 1) must keep exactly the eng group"
        );
        let row = &sol[0];
        assert_eq!(
            row.get(&Variable::new_unchecked("d")),
            Some(&iri("http://ex/eng")),
            "surviving group key must be eng"
        );
        // The synthetic aggregate column must not leak downstream.
        assert!(
            row.keys().all(|k| !k.name().starts_with("__having_agg_")),
            "synthetic HAVING aggregate variable must be stripped"
        );
    }

    /// Two subjects, each with an <a> and <b> integer value.
    fn product_dataset() -> InMemoryDataset {
        let a = iri("http://ex/a");
        let b = iri("http://ex/b");
        let two = Term::Literal(Literal {
            value: "2".to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        });
        let three = Term::Literal(Literal {
            value: "3".to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        });
        let four = Term::Literal(Literal {
            value: "4".to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        });
        let five = Term::Literal(Literal {
            value: "5".to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        });
        InMemoryDataset::from_triples(vec![
            (iri("http://ex/s1"), a.clone(), two),
            (iri("http://ex/s1"), b.clone(), three),
            (iri("http://ex/s2"), a, four),
            (iri("http://ex/s2"), b, five),
        ])
    }

    fn product_group_having(threshold: i64) -> Algebra {
        // Implicit single group over `?s <a> ?a . ?s <b> ?b`, HAVING(SUM(?a*?b) >= k).
        let bgp = Algebra::Bgp(vec![
            TriplePattern {
                subject: v("s"),
                predicate: iri("http://ex/a"),
                object: v("a"),
            },
            TriplePattern {
                subject: v("s"),
                predicate: iri("http://ex/b"),
                object: v("b"),
            },
        ]);
        let group = Algebra::Group {
            pattern: Box::new(bgp),
            variables: Vec::new(),
            aggregates: Vec::new(),
        };
        let sum_product = Expression::Function {
            name: "SUM".to_string(),
            args: vec![Expression::Binary {
                op: BinaryOperator::Multiply,
                left: Box::new(ev("a")),
                right: Box::new(ev("b")),
            }],
        };
        Algebra::Having {
            pattern: Box::new(group),
            condition: Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: Box::new(sum_product),
                right: Box::new(int_expr(threshold)),
            },
        }
    }

    #[test]
    fn having_sum_of_product_aggregate_kept_and_dropped() {
        // SUM(?a*?b) = 2*3 + 4*5 = 26.
        let exec = QueryExecutor::new();
        let ds = product_dataset();
        let kept = exec
            .execute_serial(&product_group_having(20), &ds)
            .expect("having sum kept");
        assert_eq!(kept.len(), 1, "SUM(?a*?b)=26 >= 20 must keep the group");
        let dropped = exec
            .execute_serial(&product_group_having(30), &ds)
            .expect("having sum dropped");
        assert!(
            dropped.is_empty(),
            "SUM(?a*?b)=26 >= 30 must drop the group"
        );
    }

    #[test]
    fn having_wrong_arity_aggregate_fails_loud() {
        // SUM with two arguments is a wrong-arity aggregate call; it must fail
        // loud with the arity error rather than silently mis-evaluating.
        let exec = QueryExecutor::new();
        let ds = product_dataset();
        let group = Algebra::Group {
            pattern: Box::new(bgp_s_dept_d()),
            variables: Vec::new(),
            aggregates: Vec::new(),
        };
        let algebra = Algebra::Having {
            pattern: Box::new(group),
            condition: Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: Box::new(Expression::Function {
                    name: "SUM".to_string(),
                    args: vec![ev("a"), ev("b")],
                }),
                right: Box::new(int_expr(1)),
            },
        };
        let result = exec.execute_serial(&algebra, &ds);
        let err = result.expect_err("wrong-arity SUM in HAVING must error");
        assert!(
            err.to_string()
                .contains("SUM in HAVING expects exactly one argument"),
            "arity error message must be preserved, got: {err}"
        );
    }

    #[test]
    fn filter_unknown_function_propagates_loudly() {
        // A genuinely unknown function inside FILTER must fail the whole query
        // loud (typed UnknownFunctionError), never silently drop rows to a
        // 200-with-empty result.
        use crate::executor::UnknownFunctionError;
        let exec = QueryExecutor::new();
        let ds = sample_dataset();
        let algebra = Algebra::Filter {
            pattern: Box::new(bgp_s_p_o()),
            condition: Expression::Function {
                name: "no_such_fn".to_string(),
                args: vec![ev("o")],
            },
        };
        let err = exec
            .execute_serial(&algebra, &ds)
            .expect_err("unknown function in FILTER must fail loud, not silently drop rows");
        let unknown = err
            .downcast_ref::<UnknownFunctionError>()
            .expect("error must be a typed UnknownFunctionError");
        assert_eq!(unknown.0, "no_such_fn");
    }

    #[test]
    fn having_unknown_function_propagates_loudly() {
        // The same fail-loud contract applies to HAVING row evaluation.
        use crate::executor::UnknownFunctionError;
        let exec = QueryExecutor::new();
        let ds = dept_dataset();
        let group = Algebra::Group {
            pattern: Box::new(bgp_s_dept_d()),
            variables: vec![GroupCondition {
                expr: ev("d"),
                alias: None,
            }],
            aggregates: Vec::new(),
        };
        let algebra = Algebra::Having {
            pattern: Box::new(group),
            condition: Expression::Function {
                name: "no_such_fn".to_string(),
                args: vec![ev("d")],
            },
        };
        let err = exec
            .execute_serial(&algebra, &ds)
            .expect_err("unknown function in HAVING must fail loud, not silently drop groups");
        assert!(
            err.downcast_ref::<UnknownFunctionError>().is_some(),
            "HAVING unknown-function error must be a typed UnknownFunctionError, got: {err}"
        );
    }

    #[test]
    fn filter_per_row_type_error_excludes_row_not_propagated() {
        // Two rows: one numerically comparable, one whose object is a blank node
        // (a genuine type error under `>`). SPARQL 1.1 §17.3 excludes only the
        // erroring row; the query must still succeed (guard against
        // over-propagating non-UnknownFunctionError errors).
        let exec = QueryExecutor::new();
        let mut ds = InMemoryDataset::new();
        let int5 = Term::Literal(Literal {
            value: "5".to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        });
        ds.add_triple(iri("http://ex/s1"), iri("http://ex/p"), int5);
        ds.add_triple(
            iri("http://ex/s2"),
            iri("http://ex/p"),
            Term::BlankNode("b".to_string()),
        );

        let algebra = Algebra::Filter {
            pattern: Box::new(bgp_s_p_o()),
            condition: Expression::Binary {
                op: BinaryOperator::Greater,
                left: Box::new(ev("o")),
                right: Box::new(int_expr(3)),
            },
        };
        let sol = exec
            .execute_serial(&algebra, &ds)
            .expect("a per-row type error must not fail the whole query");
        assert_eq!(
            sol.len(),
            1,
            "only the numeric row survives; the blank-node row is excluded, not propagated"
        );
        assert_eq!(
            sol[0].get(&Variable::new_unchecked("s")),
            Some(&iri("http://ex/s1"))
        );
    }

    #[test]
    fn execute_serial_table_is_join_identity() {
        let exec = QueryExecutor::new();
        let ds = InMemoryDataset::new();
        let sol = exec
            .execute_serial(&Algebra::Table, &ds)
            .expect("table serial");
        assert_eq!(sol.len(), 1, "unit table must yield one empty binding");
        assert!(sol[0].is_empty());
    }

    #[test]
    fn execute_serial_filter_in_and_not_in() {
        let exec = QueryExecutor::new();
        let ds = sample_dataset();

        // FILTER(?o IN (<o1>, <o2>)) keeps both rows.
        let in_list = Expression::Function {
            name: "list".to_string(),
            args: vec![
                Expression::Iri(NamedNode::new_unchecked("http://ex/o1")),
                Expression::Iri(NamedNode::new_unchecked("http://ex/o2")),
            ],
        };
        let in_filter = Algebra::Filter {
            pattern: Box::new(bgp_s_p_o()),
            condition: Expression::Binary {
                op: BinaryOperator::In,
                left: Box::new(Expression::Variable(Variable::new_unchecked("o"))),
                right: Box::new(in_list),
            },
        };
        let sol = exec.execute_serial(&in_filter, &ds).expect("filter in");
        assert_eq!(sol.len(), 2, "IN over both objects must keep both rows");

        // FILTER(?o NOT IN (<o1>)) keeps only the o2 row.
        let not_in_filter = Algebra::Filter {
            pattern: Box::new(bgp_s_p_o()),
            condition: Expression::Binary {
                op: BinaryOperator::NotIn,
                left: Box::new(Expression::Variable(Variable::new_unchecked("o"))),
                right: Box::new(Expression::Iri(NamedNode::new_unchecked("http://ex/o1"))),
            },
        };
        let sol2 = exec
            .execute_serial(&not_in_filter, &ds)
            .expect("filter not in");
        assert_eq!(sol2.len(), 1, "NOT IN must exclude only the listed value");
    }

    #[test]
    fn execution_budget_aborts_scan_over_triple_limit() {
        use crate::query_governor::{ExecutionBudget, ResourceBudget};

        let ds = InMemoryDataset::from_triples(vec![
            (iri("http://ex/s1"), iri("http://ex/p"), iri("http://ex/o1")),
            (iri("http://ex/s2"), iri("http://ex/p"), iri("http://ex/o2")),
            (iri("http://ex/s3"), iri("http://ex/p"), iri("http://ex/o3")),
        ]);
        let budget = ExecutionBudget::new(ResourceBudget {
            max_wall_time: None,
            max_result_rows: None,
            max_triples_scanned: Some(1),
        });
        let exec = QueryExecutor::new().with_budget(budget);
        // Scanning 3 triples under a 1-triple budget must abort in the scan hot
        // path (record_triple_scan wired into execute_single_pattern).
        let result = exec.execute_serial(&bgp_s_p_o(), &ds);
        assert!(
            result.is_err(),
            "triple-scan budget must be enforced during the BGP scan"
        );
    }
}
