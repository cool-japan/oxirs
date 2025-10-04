//! VALUES Clause Support for SPARQL
//!
//! This module provides comprehensive support for SPARQL VALUES clauses,
//! including optimization, efficient execution, and integration with joins.

use crate::algebra::{Binding, Solution, Term, Variable};
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};

/// VALUES clause representation
#[derive(Debug, Clone, PartialEq)]
pub struct ValuesClause {
    /// Variables in the VALUES clause
    pub variables: Vec<Variable>,
    /// Data rows (each row corresponds to one solution)
    pub data: Vec<Vec<Option<Term>>>,
}

impl ValuesClause {
    /// Create a new VALUES clause
    pub fn new(variables: Vec<Variable>) -> Self {
        Self {
            variables,
            data: Vec::new(),
        }
    }

    /// Create VALUES clause with data
    pub fn with_data(variables: Vec<Variable>, data: Vec<Vec<Option<Term>>>) -> Result<Self> {
        // Validate data dimensions
        for row in &data {
            if row.len() != variables.len() {
                bail!(
                    "VALUES data row has {} values but {} variables defined",
                    row.len(),
                    variables.len()
                );
            }
        }

        Ok(Self { variables, data })
    }

    /// Add a data row
    pub fn add_row(&mut self, row: Vec<Option<Term>>) -> Result<()> {
        if row.len() != self.variables.len() {
            bail!(
                "Row has {} values but {} variables defined",
                row.len(),
                self.variables.len()
            );
        }
        self.data.push(row);
        Ok(())
    }

    /// Get number of rows
    pub fn row_count(&self) -> usize {
        self.data.len()
    }

    /// Get number of variables
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Check if clause is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to solution (bindings)
    pub fn to_solution(&self) -> Solution {
        let mut solution = Solution::new();

        for row in &self.data {
            let mut binding = Binding::new();
            for (var, value) in self.variables.iter().zip(row.iter()) {
                if let Some(term) = value {
                    binding.insert(var.clone(), term.clone());
                }
                // Skip UNDEF (None) values - they don't create bindings
            }
            solution.push(binding);
        }

        solution
    }

    /// Get statistics about the VALUES clause
    pub fn statistics(&self) -> ValuesStatistics {
        let mut undef_count = 0;
        let mut distinct_values_per_var = HashMap::new();

        for var in &self.variables {
            distinct_values_per_var.insert(var.clone(), HashSet::new());
        }

        for row in &self.data {
            for (var, value) in self.variables.iter().zip(row.iter()) {
                if let Some(term) = value {
                    distinct_values_per_var
                        .get_mut(var)
                        .unwrap()
                        .insert(term.clone());
                } else {
                    undef_count += 1;
                }
            }
        }

        let selectivity: HashMap<Variable, f64> = distinct_values_per_var
            .iter()
            .map(|(var, values)| {
                let selectivity = if self.data.is_empty() {
                    1.0
                } else {
                    values.len() as f64 / self.data.len() as f64
                };
                (var.clone(), selectivity)
            })
            .collect();

        ValuesStatistics {
            row_count: self.data.len(),
            variable_count: self.variables.len(),
            undef_count,
            selectivity,
            estimated_memory: self.estimate_memory_usage(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        // Rough estimate: 32 bytes per term + overhead
        self.data.len() * self.variables.len() * 32
    }
}

/// Statistics about a VALUES clause
#[derive(Debug, Clone)]
pub struct ValuesStatistics {
    pub row_count: usize,
    pub variable_count: usize,
    pub undef_count: usize,
    pub selectivity: HashMap<Variable, f64>,
    pub estimated_memory: usize,
}

/// VALUES clause optimizer
pub struct ValuesOptimizer;

impl ValuesOptimizer {
    /// Optimize VALUES clause
    pub fn optimize(values: &ValuesClause) -> OptimizedValues {
        let stats = values.statistics();

        // Determine optimal execution strategy
        let strategy = if values.row_count() <= 10 {
            ValuesExecutionStrategy::Inline
        } else if values.row_count() <= 1000 {
            ValuesExecutionStrategy::HashIndex
        } else {
            ValuesExecutionStrategy::Materialized
        };

        // Check if VALUES can be pushed into joins
        let pushable = values.row_count() < 100 && stats.undef_count == 0;

        // Identify most selective variable for indexing
        let most_selective = stats
            .selectivity
            .iter()
            .min_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(var, _)| var.clone());

        OptimizedValues {
            original: values.clone(),
            strategy,
            pushable,
            most_selective_var: most_selective,
            statistics: stats,
        }
    }

    /// Suggest reordering for VALUES clause
    pub fn suggest_variable_order(values: &ValuesClause) -> Vec<Variable> {
        let stats = values.statistics();

        // Order variables by selectivity (most selective first)
        let mut vars_with_selectivity: Vec<_> = stats
            .selectivity
            .iter()
            .map(|(var, sel)| (var.clone(), *sel))
            .collect();

        vars_with_selectivity.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        vars_with_selectivity
            .into_iter()
            .map(|(var, _)| var)
            .collect()
    }
}

/// Optimized VALUES representation
#[derive(Debug, Clone)]
pub struct OptimizedValues {
    pub original: ValuesClause,
    pub strategy: ValuesExecutionStrategy,
    pub pushable: bool,
    pub most_selective_var: Option<Variable>,
    pub statistics: ValuesStatistics,
}

/// VALUES execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValuesExecutionStrategy {
    /// Inline values directly into query
    Inline,
    /// Build hash index for efficient lookups
    HashIndex,
    /// Materialize as temporary table
    Materialized,
}

/// Hash-indexed VALUES for efficient joins
pub struct IndexedValues {
    values: ValuesClause,
    /// Index: Variable -> Term -> List of row indices
    indexes: HashMap<Variable, HashMap<Term, Vec<usize>>>,
}

impl IndexedValues {
    /// Create indexed VALUES
    pub fn new(values: ValuesClause) -> Self {
        let mut indexes = HashMap::new();

        for (var_idx, var) in values.variables.iter().enumerate() {
            let mut var_index: HashMap<Term, Vec<usize>> = HashMap::new();

            for (row_idx, row) in values.data.iter().enumerate() {
                if let Some(Some(term)) = row.get(var_idx) {
                    var_index
                        .entry(term.clone())
                        .or_insert_with(Vec::new)
                        .push(row_idx);
                }
            }

            indexes.insert(var.clone(), var_index);
        }

        Self { values, indexes }
    }

    /// Lookup rows by variable binding
    pub fn lookup(&self, var: &Variable, term: &Term) -> Vec<Binding> {
        if let Some(var_index) = self.indexes.get(var) {
            if let Some(row_indices) = var_index.get(term) {
                return row_indices
                    .iter()
                    .filter_map(|&idx| self.row_to_binding(idx))
                    .collect();
            }
        }
        Vec::new()
    }

    /// Probe with multiple variable bindings
    pub fn probe(&self, binding: &Binding) -> Vec<Binding> {
        let mut candidates: Option<HashSet<usize>> = None;

        // Find candidate rows by intersecting results from each bound variable
        for (var, term) in binding {
            if let Some(var_index) = self.indexes.get(var) {
                if let Some(row_indices) = var_index.get(term) {
                    let row_set: HashSet<usize> = row_indices.iter().copied().collect();

                    candidates = Some(if let Some(existing) = candidates {
                        existing.intersection(&row_set).copied().collect()
                    } else {
                        row_set
                    });
                }
            }
        }

        // Convert candidate rows to bindings
        if let Some(candidate_rows) = candidates {
            candidate_rows
                .into_iter()
                .filter_map(|idx| self.row_to_binding(idx))
                .collect()
        } else {
            // No bound variables in common, return all rows
            (0..self.values.data.len())
                .filter_map(|idx| self.row_to_binding(idx))
                .collect()
        }
    }

    fn row_to_binding(&self, row_idx: usize) -> Option<Binding> {
        let row = self.values.data.get(row_idx)?;
        let mut binding = Binding::new();

        for (var, value) in self.values.variables.iter().zip(row.iter()) {
            if let Some(term) = value {
                binding.insert(var.clone(), term.clone());
            }
        }

        Some(binding)
    }

    /// Get all bindings
    pub fn all_bindings(&self) -> Vec<Binding> {
        (0..self.values.data.len())
            .filter_map(|idx| self.row_to_binding(idx))
            .collect()
    }

    /// Get statistics
    pub fn statistics(&self) -> ValuesStatistics {
        self.values.statistics()
    }
}

/// VALUES-aware join optimizer
pub struct ValuesJoinOptimizer;

impl ValuesJoinOptimizer {
    /// Optimize join with VALUES clause
    pub fn optimize_join(
        values: &ValuesClause,
        other_variables: &HashSet<Variable>,
    ) -> JoinStrategy {
        let _stats = values.statistics();

        // Find shared variables
        let values_vars: HashSet<_> = values.variables.iter().collect();
        let shared_vars: Vec<_> = values_vars
            .intersection(
                &other_variables
                    .iter()
                    .collect::<HashSet<_>>(),
            )
            .map(|&v| v.clone())
            .collect();

        if shared_vars.is_empty() {
            return JoinStrategy::CrossProduct;
        }

        // Choose strategy based on VALUES size and shared variables
        if values.row_count() <= 10 {
            JoinStrategy::NestedLoop
        } else if shared_vars.len() == 1 {
            JoinStrategy::IndexNestedLoop {
                index_var: shared_vars[0].clone(),
            }
        } else {
            JoinStrategy::HashJoin {
                join_vars: shared_vars,
            }
        }
    }

    /// Push VALUES into join to reduce intermediate results
    pub fn can_push_values(
        values: &ValuesClause,
        join_vars: &[Variable],
    ) -> bool {
        // VALUES can be pushed if:
        // 1. It's small (< 100 rows)
        // 2. Has no UNDEF values in join variables
        // 3. Shares variables with the join

        if values.row_count() >= 100 {
            return false;
        }

        let stats = values.statistics();
        if stats.undef_count > 0 {
            // Would need to check specifically which variables have UNDEF
            return false;
        }

        // Check if any VALUES variables are in the join
        values
            .variables
            .iter()
            .any(|v| join_vars.contains(v))
    }
}

/// Join strategy for VALUES
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinStrategy {
    CrossProduct,
    NestedLoop,
    IndexNestedLoop { index_var: Variable },
    HashJoin { join_vars: Vec<Variable> },
}

/// VALUES clause executor
pub struct ValuesExecutor {
    indexed_cache: HashMap<String, IndexedValues>,
}

impl ValuesExecutor {
    /// Create new VALUES executor
    pub fn new() -> Self {
        Self {
            indexed_cache: HashMap::new(),
        }
    }

    /// Execute VALUES clause
    pub fn execute(&mut self, values: &ValuesClause) -> Result<Solution> {
        Ok(values.to_solution())
    }

    /// Execute VALUES clause with indexing
    pub fn execute_indexed(&mut self, values: &ValuesClause) -> Result<IndexedValues> {
        let cache_key = self.compute_cache_key(values);

        if let Some(indexed) = self.indexed_cache.get(&cache_key) {
            return Ok(IndexedValues {
                values: indexed.values.clone(),
                indexes: indexed.indexes.clone(),
            });
        }

        let indexed = IndexedValues::new(values.clone());
        self.indexed_cache
            .insert(cache_key, indexed.clone_structure());

        Ok(indexed)
    }

    /// Join VALUES with solution using index
    pub fn join_indexed(
        &mut self,
        values: &ValuesClause,
        solution: Solution,
    ) -> Result<Solution> {
        let indexed = self.execute_indexed(values)?;
        let mut result = Solution::new();

        for binding in solution {
            // Probe indexed VALUES with current binding
            let matches = indexed.probe(&binding);

            for values_binding in &matches {
                // Merge bindings
                let mut merged = binding.clone();
                for (var, term) in values_binding {
                    if let Some(existing) = merged.get(var) {
                        if existing != term {
                            // Conflict - skip this combination
                            continue;
                        }
                    } else {
                        merged.insert(var.clone(), term.clone());
                    }
                }
                result.push(merged);
            }

            // If no matches, it's a cross product (no shared variables)
            if matches.is_empty() && !has_shared_vars(values, &binding) {
                result.push(binding);
            }
        }

        Ok(result)
    }

    fn compute_cache_key(&self, values: &ValuesClause) -> String {
        format!("{:?}", values)
    }
}

impl Default for ValuesExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexedValues {
    fn clone_structure(&self) -> Self {
        Self {
            values: self.values.clone(),
            indexes: self.indexes.clone(),
        }
    }
}

/// Check if VALUES and binding have shared variables
fn has_shared_vars(values: &ValuesClause, binding: &Binding) -> bool {
    values
        .variables
        .iter()
        .any(|v| binding.contains_key(v))
}

/// VALUES clause builder for programmatic construction
pub struct ValuesBuilder {
    variables: Vec<Variable>,
    rows: Vec<Vec<Option<Term>>>,
}

impl ValuesBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// Add a variable
    pub fn add_variable(mut self, var: Variable) -> Self {
        self.variables.push(var);
        self
    }

    /// Add multiple variables
    pub fn add_variables(mut self, vars: Vec<Variable>) -> Self {
        self.variables.extend(vars);
        self
    }

    /// Add a row of values
    pub fn add_row(mut self, values: Vec<Option<Term>>) -> Result<Self> {
        if values.len() != self.variables.len() {
            bail!(
                "Row has {} values but {} variables defined",
                values.len(),
                self.variables.len()
            );
        }
        self.rows.push(values);
        Ok(self)
    }

    /// Build the VALUES clause
    pub fn build(self) -> Result<ValuesClause> {
        ValuesClause::with_data(self.variables, self.rows)
    }
}

impl Default for ValuesBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::NamedNode;

    fn create_test_variable(name: &str) -> Variable {
        Variable::new(name).unwrap()
    }

    fn create_test_iri(iri: &str) -> Term {
        Term::Iri(NamedNode::new(iri).unwrap())
    }

    #[test]
    fn test_values_clause_creation() {
        let var_x = create_test_variable("x");
        let var_y = create_test_variable("y");

        let mut values = ValuesClause::new(vec![var_x.clone(), var_y.clone()]);

        assert_eq!(values.variable_count(), 2);
        assert_eq!(values.row_count(), 0);
        assert!(values.is_empty());

        values
            .add_row(vec![
                Some(create_test_iri("http://example.org/a")),
                Some(create_test_iri("http://example.org/b")),
            ])
            .unwrap();

        assert_eq!(values.row_count(), 1);
        assert!(!values.is_empty());
    }

    #[test]
    fn test_values_with_undef() {
        let var_x = create_test_variable("x");
        let var_y = create_test_variable("y");

        let values = ValuesClause::with_data(
            vec![var_x.clone(), var_y.clone()],
            vec![
                vec![
                    Some(create_test_iri("http://example.org/a")),
                    Some(create_test_iri("http://example.org/b")),
                ],
                vec![
                    Some(create_test_iri("http://example.org/c")),
                    None, // UNDEF
                ],
            ],
        )
        .unwrap();

        let solution = values.to_solution();
        assert_eq!(solution.len(), 2);

        // First binding should have both variables
        assert_eq!(solution[0].len(), 2);

        // Second binding should only have x (y is UNDEF)
        assert_eq!(solution[1].len(), 1);
        assert!(solution[1].contains_key(&var_x));
        assert!(!solution[1].contains_key(&var_y));
    }

    #[test]
    fn test_values_statistics() {
        let var_x = create_test_variable("x");
        let values = ValuesClause::with_data(
            vec![var_x.clone()],
            vec![
                vec![Some(create_test_iri("http://example.org/a"))],
                vec![Some(create_test_iri("http://example.org/a"))],
                vec![Some(create_test_iri("http://example.org/b"))],
            ],
        )
        .unwrap();

        let stats = values.statistics();
        assert_eq!(stats.row_count, 3);
        assert_eq!(stats.variable_count, 1);
        assert_eq!(stats.undef_count, 0);

        // Selectivity should be 2/3 (2 distinct values, 3 rows)
        let selectivity = stats.selectivity.get(&var_x).unwrap();
        assert!((selectivity - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_values_optimizer() {
        let var_x = create_test_variable("x");
        let values = ValuesClause::with_data(
            vec![var_x],
            vec![
                vec![Some(create_test_iri("http://example.org/a"))],
                vec![Some(create_test_iri("http://example.org/b"))],
            ],
        )
        .unwrap();

        let optimized = ValuesOptimizer::optimize(&values);
        assert_eq!(optimized.strategy, ValuesExecutionStrategy::Inline);
        assert!(optimized.pushable);
    }

    #[test]
    fn test_indexed_values() {
        let var_x = create_test_variable("x");
        let var_y = create_test_variable("y");

        let values = ValuesClause::with_data(
            vec![var_x.clone(), var_y.clone()],
            vec![
                vec![
                    Some(create_test_iri("http://example.org/a")),
                    Some(create_test_iri("http://example.org/1")),
                ],
                vec![
                    Some(create_test_iri("http://example.org/b")),
                    Some(create_test_iri("http://example.org/2")),
                ],
            ],
        )
        .unwrap();

        let indexed = IndexedValues::new(values);

        // Lookup by x
        let results = indexed.lookup(&var_x, &create_test_iri("http://example.org/a"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 2);

        // Get all bindings
        let all = indexed.all_bindings();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_indexed_values_probe() {
        let var_x = create_test_variable("x");
        let var_y = create_test_variable("y");

        let values = ValuesClause::with_data(
            vec![var_x.clone(), var_y.clone()],
            vec![
                vec![
                    Some(create_test_iri("http://example.org/a")),
                    Some(create_test_iri("http://example.org/1")),
                ],
                vec![
                    Some(create_test_iri("http://example.org/a")),
                    Some(create_test_iri("http://example.org/2")),
                ],
            ],
        )
        .unwrap();

        let indexed = IndexedValues::new(values);

        // Probe with partial binding
        let mut probe_binding = Binding::new();
        probe_binding.insert(var_x.clone(), create_test_iri("http://example.org/a"));

        let results = indexed.probe(&probe_binding);
        assert_eq!(results.len(), 2); // Both rows have x=a
    }

    #[test]
    fn test_values_join_optimizer() {
        let var_x = create_test_variable("x");
        // Create VALUES with > 10 rows to trigger IndexNestedLoop strategy
        let data: Vec<Vec<Option<Term>>> = (0..15)
            .map(|i| {
                vec![Some(create_test_iri(&format!(
                    "http://example.org/item{}",
                    i
                )))]
            })
            .collect();
        let values = ValuesClause::with_data(vec![var_x.clone()], data).unwrap();

        let other_vars = [var_x.clone()].into_iter().collect();

        let strategy = ValuesJoinOptimizer::optimize_join(&values, &other_vars);
        assert_eq!(
            strategy,
            JoinStrategy::IndexNestedLoop {
                index_var: var_x
            }
        );
    }

    #[test]
    fn test_values_executor() {
        let var_x = create_test_variable("x");
        let values = ValuesClause::with_data(
            vec![var_x],
            vec![
                vec![Some(create_test_iri("http://example.org/a"))],
                vec![Some(create_test_iri("http://example.org/b"))],
            ],
        )
        .unwrap();

        let mut executor = ValuesExecutor::new();
        let solution = executor.execute(&values).unwrap();

        assert_eq!(solution.len(), 2);
    }

    #[test]
    fn test_values_builder() {
        let var_x = create_test_variable("x");
        let var_y = create_test_variable("y");

        let values = ValuesBuilder::new()
            .add_variable(var_x)
            .add_variable(var_y)
            .add_row(vec![
                Some(create_test_iri("http://example.org/a")),
                Some(create_test_iri("http://example.org/1")),
            ])
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(values.variable_count(), 2);
        assert_eq!(values.row_count(), 1);
    }

    #[test]
    fn test_values_invalid_row() {
        let var_x = create_test_variable("x");
        let mut values = ValuesClause::new(vec![var_x]);

        // Try to add row with wrong number of values
        let result = values.add_row(vec![
            Some(create_test_iri("http://example.org/a")),
            Some(create_test_iri("http://example.org/b")),
        ]);

        assert!(result.is_err());
    }

    #[test]
    fn test_variable_reordering() {
        let var_x = create_test_variable("x");
        let var_y = create_test_variable("y");

        let values = ValuesClause::with_data(
            vec![var_x.clone(), var_y.clone()],
            vec![
                vec![
                    Some(create_test_iri("http://example.org/a")),
                    Some(create_test_iri("http://example.org/1")),
                ],
                vec![
                    Some(create_test_iri("http://example.org/a")),
                    Some(create_test_iri("http://example.org/2")),
                ],
                vec![
                    Some(create_test_iri("http://example.org/b")),
                    Some(create_test_iri("http://example.org/3")),
                ],
            ],
        )
        .unwrap();

        let ordered = ValuesOptimizer::suggest_variable_order(&values);
        assert_eq!(ordered.len(), 2);
        // x should be first (more selective: 2 distinct values vs 3)
        assert_eq!(ordered[0], var_x);
    }

    #[test]
    fn test_can_push_values() {
        let var_x = create_test_variable("x");
        let values = ValuesClause::with_data(
            vec![var_x.clone()],
            vec![vec![Some(create_test_iri("http://example.org/a"))]],
        )
        .unwrap();

        assert!(ValuesJoinOptimizer::can_push_values(&values, &[var_x]));
    }

    #[test]
    fn test_large_values_strategy() {
        let var_x = create_test_variable("x");
        let mut data = Vec::new();
        for i in 0..2000 {
            data.push(vec![Some(create_test_iri(&format!(
                "http://example.org/{}",
                i
            )))]);
        }

        let values = ValuesClause::with_data(vec![var_x], data).unwrap();
        let optimized = ValuesOptimizer::optimize(&values);

        assert_eq!(optimized.strategy, ValuesExecutionStrategy::Materialized);
        assert!(!optimized.pushable); // Too large to push
    }
}
