//! # Advanced Query Execution Engine
//!
//! Sophisticated query execution interface with pattern matching, variable binding,
//! join optimization, and iterator-based result streaming for efficient SPARQL-like queries.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::mvcc::TransactionId;
use crate::nodes::NodeId;
use crate::triple_store::{Triple, TripleStore};

/// Variable in a query pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
}

impl Variable {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

/// Query pattern element that can be bound or variable
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternElement {
    /// Bound to a specific node ID
    Bound(NodeId),
    /// Unbound variable that can match any value
    Variable(Variable),
    /// Wildcard that matches any value but doesn't bind
    Any,
}

impl PatternElement {
    /// Check if this element is bound to a specific value
    pub fn is_bound(&self) -> bool {
        matches!(self, PatternElement::Bound(_))
    }

    /// Check if this element is a variable that can be bound
    pub fn is_variable(&self) -> bool {
        matches!(self, PatternElement::Variable(_))
    }

    /// Get the variable name if this is a variable
    pub fn variable_name(&self) -> Option<&str> {
        match self {
            PatternElement::Variable(var) => Some(&var.name),
            _ => None,
        }
    }

    /// Get the bound value if this element is bound
    pub fn bound_value(&self) -> Option<NodeId> {
        match self {
            PatternElement::Bound(id) => Some(*id),
            _ => None,
        }
    }
}

/// Triple pattern with optional variables
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
}

impl TriplePattern {
    /// Create a new triple pattern
    pub fn new(subject: PatternElement, predicate: PatternElement, object: PatternElement) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Get all variables in this pattern
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();

        if let PatternElement::Variable(var) = &self.subject {
            vars.push(var.clone());
        }
        if let PatternElement::Variable(var) = &self.predicate {
            vars.push(var.clone());
        }
        if let PatternElement::Variable(var) = &self.object {
            vars.push(var.clone());
        }

        vars
    }

    /// Get the selectivity score (lower is more selective)
    pub fn selectivity_score(&self) -> u32 {
        let mut score = 0;

        if !self.subject.is_bound() {
            score += 1;
        }
        if !self.predicate.is_bound() {
            score += 1;
        }
        if !self.object.is_bound() {
            score += 1;
        }

        score
    }

    /// Check if pattern matches a triple with given variable bindings
    pub fn matches(&self, triple: &Triple, bindings: &VariableBindings) -> bool {
        self.matches_element(&self.subject, triple.subject, bindings)
            && self.matches_element(&self.predicate, triple.predicate, bindings)
            && self.matches_element(&self.object, triple.object, bindings)
    }

    fn matches_element(
        &self,
        pattern: &PatternElement,
        value: NodeId,
        bindings: &VariableBindings,
    ) -> bool {
        match pattern {
            PatternElement::Bound(bound_value) => *bound_value == value,
            PatternElement::Variable(var) => {
                if let Some(bound_value) = bindings.get(&var.name) {
                    *bound_value == value
                } else {
                    true // Unbound variable matches any value
                }
            }
            PatternElement::Any => true,
        }
    }

    /// Convert to a query signature for index selection
    pub fn to_query_signature(&self) -> (Option<NodeId>, Option<NodeId>, Option<NodeId>) {
        (
            self.subject.bound_value(),
            self.predicate.bound_value(),
            self.object.bound_value(),
        )
    }
}

/// Variable bindings for query execution
pub type VariableBindings = HashMap<String, NodeId>;

/// Solution binding for a query result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolutionBinding {
    pub bindings: VariableBindings,
}

impl SolutionBinding {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    pub fn with_binding(mut self, var: String, value: NodeId) -> Self {
        self.bindings.insert(var, value);
        self
    }

    pub fn merge(&self, other: &SolutionBinding) -> Option<SolutionBinding> {
        let mut merged = self.bindings.clone();

        for (var, value) in &other.bindings {
            if let Some(existing_value) = merged.get(var) {
                if existing_value != value {
                    return None; // Conflict in variable binding
                }
            } else {
                merged.insert(var.clone(), *value);
            }
        }

        Some(SolutionBinding { bindings: merged })
    }

    pub fn get(&self, var: &str) -> Option<NodeId> {
        self.bindings.get(var).copied()
    }

    pub fn contains_var(&self, var: &str) -> bool {
        self.bindings.contains_key(var)
    }
}

/// Query execution plan
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub patterns: Vec<TriplePattern>,
    pub join_order: Vec<usize>,
    pub estimated_cost: f64,
}

impl QueryPlan {
    pub fn new(patterns: Vec<TriplePattern>) -> Self {
        let join_order = Self::optimize_join_order(&patterns);
        let estimated_cost = Self::estimate_cost(&patterns, &join_order);

        Self {
            patterns,
            join_order,
            estimated_cost,
        }
    }

    /// Optimize join order based on selectivity
    fn optimize_join_order(patterns: &[TriplePattern]) -> Vec<usize> {
        let mut indexed_patterns: Vec<(usize, &TriplePattern)> =
            patterns.iter().enumerate().collect();

        // Sort by selectivity (most selective first)
        indexed_patterns.sort_by_key(|(_, pattern)| pattern.selectivity_score());

        indexed_patterns.into_iter().map(|(i, _)| i).collect()
    }

    /// Estimate execution cost
    fn estimate_cost(_patterns: &[TriplePattern], _join_order: &[usize]) -> f64 {
        // Simplified cost model - in practice this would use statistics
        100.0
    }
}

/// Advanced query execution engine
pub struct QueryExecutor {
    store: Arc<TripleStore>,
    statistics: Arc<Mutex<QueryStatistics>>,
}

/// Query execution statistics
#[derive(Debug, Clone, Default)]
pub struct QueryStatistics {
    pub total_queries: u64,
    pub total_execution_time_ms: u64,
    pub total_results_returned: u64,
    pub index_scan_count: u64,
    pub join_count: u64,
    pub avg_execution_time_ms: f64,
}

impl QueryStatistics {
    pub fn record_query(&mut self, execution_time_ms: u64, results_count: u64) {
        self.total_queries += 1;
        self.total_execution_time_ms += execution_time_ms;
        self.total_results_returned += results_count;
        self.avg_execution_time_ms =
            self.total_execution_time_ms as f64 / self.total_queries as f64;
    }
}

impl QueryExecutor {
    /// Create a new query executor
    pub fn new(store: Arc<TripleStore>) -> Self {
        Self {
            store,
            statistics: Arc::new(Mutex::new(QueryStatistics::default())),
        }
    }

    /// Execute a single triple pattern query
    pub fn execute_pattern(
        &self,
        tx_id: TransactionId,
        pattern: &TriplePattern,
    ) -> Result<QueryResultIterator> {
        let start_time = std::time::Instant::now();

        let signature = pattern.to_query_signature();
        let results = self
            .store
            .query_triples_tx(tx_id, signature.0, signature.1, signature.2)?;

        // Convert results to solution bindings
        let mut solutions = Vec::new();
        let _variables = pattern.variables();

        for triple in results {
            let mut binding = SolutionBinding::new();

            // Bind variables based on pattern
            if let PatternElement::Variable(var) = &pattern.subject {
                binding = binding.with_binding(var.name.clone(), triple.subject);
            }
            if let PatternElement::Variable(var) = &pattern.predicate {
                binding = binding.with_binding(var.name.clone(), triple.predicate);
            }
            if let PatternElement::Variable(var) = &pattern.object {
                binding = binding.with_binding(var.name.clone(), triple.object);
            }

            solutions.push(binding);
        }

        let execution_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        if let Ok(mut stats) = self.statistics.lock() {
            stats.record_query(execution_time, solutions.len() as u64);
            stats.index_scan_count += 1;
        }

        Ok(QueryResultIterator::new(solutions))
    }

    /// Execute a multi-pattern query with joins
    pub fn execute_query(
        &self,
        tx_id: TransactionId,
        patterns: Vec<TriplePattern>,
    ) -> Result<QueryResultIterator> {
        let start_time = std::time::Instant::now();

        if patterns.is_empty() {
            return Ok(QueryResultIterator::new(Vec::new()));
        }

        // Create and optimize query plan
        let plan = QueryPlan::new(patterns);

        // Execute query plan
        let mut intermediate_results: Option<Vec<SolutionBinding>> = None;

        for &pattern_idx in &plan.join_order {
            let pattern = &plan.patterns[pattern_idx];

            if intermediate_results.is_none() {
                // First pattern - get initial results
                let pattern_results = self.execute_pattern(tx_id, pattern)?;
                intermediate_results = Some(pattern_results.collect());
            } else {
                // Subsequent patterns - join with existing results
                let current_results = intermediate_results.take().unwrap();
                let mut joined_results = Vec::new();

                for binding in current_results {
                    // Apply current bindings to pattern
                    let specialized_pattern = self.apply_bindings_to_pattern(pattern, &binding);

                    // Execute specialized pattern
                    let pattern_results = self.execute_pattern(tx_id, &specialized_pattern)?;

                    // Join results
                    for pattern_binding in pattern_results.collect() {
                        if let Some(merged) = binding.merge(&pattern_binding) {
                            joined_results.push(merged);
                        }
                    }
                }

                intermediate_results = Some(joined_results);

                // Update join statistics
                if let Ok(mut stats) = self.statistics.lock() {
                    stats.join_count += 1;
                }
            }
        }

        let results = intermediate_results.unwrap_or_default();

        let execution_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        if let Ok(mut stats) = self.statistics.lock() {
            stats.record_query(execution_time, results.len() as u64);
        }

        Ok(QueryResultIterator::new(results))
    }

    /// Apply variable bindings to a pattern to create a more specific pattern
    fn apply_bindings_to_pattern(
        &self,
        pattern: &TriplePattern,
        bindings: &SolutionBinding,
    ) -> TriplePattern {
        let subject = match &pattern.subject {
            PatternElement::Variable(var) => {
                if let Some(value) = bindings.get(&var.name) {
                    PatternElement::Bound(value)
                } else {
                    pattern.subject.clone()
                }
            }
            _ => pattern.subject.clone(),
        };

        let predicate = match &pattern.predicate {
            PatternElement::Variable(var) => {
                if let Some(value) = bindings.get(&var.name) {
                    PatternElement::Bound(value)
                } else {
                    pattern.predicate.clone()
                }
            }
            _ => pattern.predicate.clone(),
        };

        let object = match &pattern.object {
            PatternElement::Variable(var) => {
                if let Some(value) = bindings.get(&var.name) {
                    PatternElement::Bound(value)
                } else {
                    pattern.object.clone()
                }
            }
            _ => pattern.object.clone(),
        };

        TriplePattern::new(subject, predicate, object)
    }

    /// Get query execution statistics
    pub fn get_statistics(&self) -> Result<QueryStatistics> {
        self.statistics
            .lock()
            .map(|stats| stats.clone())
            .map_err(|_| anyhow!("Failed to acquire statistics lock"))
    }

    /// Reset query statistics
    pub fn reset_statistics(&self) -> Result<()> {
        self.statistics
            .lock()
            .map(|mut stats| *stats = QueryStatistics::default())
            .map_err(|_| anyhow!("Failed to acquire statistics lock"))
    }
}

/// Iterator for query results with streaming support
pub struct QueryResultIterator {
    results: VecDeque<SolutionBinding>,
    position: usize,
}

impl QueryResultIterator {
    pub fn new(results: Vec<SolutionBinding>) -> Self {
        Self {
            results: results.into(),
            position: 0,
        }
    }

    pub fn collect(self) -> Vec<SolutionBinding> {
        self.results.into()
    }

    pub fn count(&self) -> usize {
        self.results.len()
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

impl Iterator for QueryResultIterator {
    type Item = SolutionBinding;

    fn next(&mut self) -> Option<Self::Item> {
        self.results.pop_front().map(|binding| {
            self.position += 1;
            binding
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.results.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for QueryResultIterator {
    fn len(&self) -> usize {
        self.results.len()
    }
}

/// Builder for creating query patterns
pub struct QueryBuilder {
    patterns: Vec<TriplePattern>,
}

impl QueryBuilder {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a triple pattern to the query
    pub fn add_pattern(mut self, pattern: TriplePattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Add a pattern with subject, predicate, object (convenience method)
    pub fn add_triple(
        mut self,
        subject: PatternElement,
        predicate: PatternElement,
        object: PatternElement,
    ) -> Self {
        self.patterns
            .push(TriplePattern::new(subject, predicate, object));
        self
    }

    /// Build the final query patterns
    pub fn build(self) -> Vec<TriplePattern> {
        self.patterns
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for creating pattern elements
pub mod pattern {
    use super::*;

    /// Create a bound pattern element
    pub fn bound(node_id: NodeId) -> PatternElement {
        PatternElement::Bound(node_id)
    }

    /// Create a variable pattern element
    pub fn var(name: &str) -> PatternElement {
        PatternElement::Variable(Variable::new(name.to_string()))
    }

    /// Create a wildcard pattern element
    pub fn any() -> PatternElement {
        PatternElement::Any
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triple_store::TripleStoreConfig;
    use tempfile::TempDir;

    fn create_test_store() -> (Arc<TripleStore>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = TripleStoreConfig {
            storage_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let store = Arc::new(TripleStore::with_config(config).unwrap());
        (store, temp_dir)
    }

    #[test]
    fn test_pattern_element_operations() {
        let bound = PatternElement::Bound(123);
        let var = PatternElement::Variable(Variable::new("x".to_string()));
        let any = PatternElement::Any;

        assert!(bound.is_bound());
        assert!(!bound.is_variable());
        assert_eq!(bound.bound_value(), Some(123));

        assert!(!var.is_bound());
        assert!(var.is_variable());
        assert_eq!(var.variable_name(), Some("x"));

        assert!(!any.is_bound());
        assert!(!any.is_variable());
    }

    #[test]
    fn test_triple_pattern_variables() {
        let pattern = TriplePattern::new(pattern::var("s"), pattern::bound(123), pattern::var("o"));

        let vars = pattern.variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.iter().any(|v| v.name == "s"));
        assert!(vars.iter().any(|v| v.name == "o"));
    }

    #[test]
    fn test_solution_binding_merge() {
        let binding1 = SolutionBinding::new().with_binding("x".to_string(), 123);
        let binding2 = SolutionBinding::new().with_binding("y".to_string(), 456);
        let binding3 = SolutionBinding::new().with_binding("x".to_string(), 789); // Conflict

        let merged = binding1.merge(&binding2);
        assert!(merged.is_some());
        let merged = merged.unwrap();
        assert_eq!(merged.get("x"), Some(123));
        assert_eq!(merged.get("y"), Some(456));

        let conflict = binding1.merge(&binding3);
        assert!(conflict.is_none()); // Should fail due to conflict
    }

    #[test]
    fn test_query_builder() {
        let patterns = QueryBuilder::new()
            .add_triple(pattern::var("s"), pattern::bound(123), pattern::var("o"))
            .add_triple(pattern::var("s"), pattern::bound(456), pattern::any())
            .build();

        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].selectivity_score(), 2); // Two variables
        assert_eq!(patterns[1].selectivity_score(), 2); // One variable + one any
    }

    #[test]
    fn test_query_plan_optimization() {
        let patterns = vec![
            TriplePattern::new(pattern::any(), pattern::any(), pattern::any()), // Least selective
            TriplePattern::new(pattern::bound(1), pattern::bound(2), pattern::bound(3)), // Most selective
            TriplePattern::new(pattern::var("x"), pattern::bound(4), pattern::var("y")), // Medium selective
        ];

        let plan = QueryPlan::new(patterns);

        // Should order by selectivity: most selective first
        assert_eq!(plan.join_order[0], 1); // All bound pattern
        assert_eq!(plan.join_order[1], 2); // One bound pattern
        assert_eq!(plan.join_order[2], 0); // All unbound pattern
    }
}
