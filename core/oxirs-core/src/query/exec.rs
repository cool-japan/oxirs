//! Query execution engine
//!
//! This module executes query plans against RDF stores.

use crate::model::*;
use crate::query::algebra::*;
use crate::query::plan::ExecutionPlan;
use crate::store::Store;
use crate::OxirsError;
use std::collections::{HashMap, HashSet};

/// A solution mapping (binding of variables to values)
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    bindings: HashMap<Variable, Term>,
}

impl Solution {
    /// Creates a new empty solution
    pub fn new() -> Self {
        Solution {
            bindings: HashMap::new(),
        }
    }

    /// Binds a variable to a value
    pub fn bind(&mut self, var: Variable, value: Term) {
        self.bindings.insert(var, value);
    }

    /// Gets the value bound to a variable
    pub fn get(&self, var: &Variable) -> Option<&Term> {
        self.bindings.get(var)
    }

    /// Merges two solutions (for joins)
    pub fn merge(&self, other: &Solution) -> Option<Solution> {
        let mut merged = self.clone();

        for (var, value) in &other.bindings {
            if let Some(existing) = merged.bindings.get(var) {
                if existing != value {
                    return None; // Incompatible bindings
                }
            } else {
                merged.bindings.insert(var.clone(), value.clone());
            }
        }

        Some(merged)
    }

    /// Projects specific variables
    pub fn project(&self, vars: &[Variable]) -> Solution {
        let mut projected = Solution::new();
        for var in vars {
            if let Some(value) = self.bindings.get(var) {
                projected.bind(var.clone(), value.clone());
            }
        }
        projected
    }
}

/// Query results
#[derive(Debug)]
pub enum QueryResults {
    /// Boolean result (for ASK queries)
    Boolean(bool),
    /// Solutions (for SELECT queries)
    Solutions(Vec<Solution>),
    /// Graph (for CONSTRUCT queries)
    Graph(Vec<Triple>),
}

/// Query executor
pub struct QueryExecutor<'a> {
    store: &'a Store,
}

impl<'a> QueryExecutor<'a> {
    /// Creates a new query executor
    pub fn new(store: &'a Store) -> Self {
        QueryExecutor { store }
    }

    /// Executes a query plan
    pub fn execute(&self, plan: &ExecutionPlan) -> Result<Vec<Solution>, OxirsError> {
        self.execute_plan(plan)
    }

    fn execute_plan(&self, plan: &ExecutionPlan) -> Result<Vec<Solution>, OxirsError> {
        match plan {
            ExecutionPlan::TripleScan { pattern } => self.execute_triple_scan(pattern),
            ExecutionPlan::HashJoin {
                left,
                right,
                join_vars,
            } => self.execute_hash_join(left, right, join_vars),
            ExecutionPlan::Filter { input, condition } => self.execute_filter(input, condition),
            ExecutionPlan::Project { input, vars } => self.execute_project(input, vars),
            ExecutionPlan::Sort { input, order_by } => self.execute_sort(input, order_by),
            ExecutionPlan::Limit {
                input,
                limit,
                offset,
            } => self.execute_limit(input, *limit, *offset),
            ExecutionPlan::Union { left, right } => self.execute_union(left, right),
            ExecutionPlan::Distinct { input } => self.execute_distinct(input),
        }
    }

    fn execute_triple_scan(&self, pattern: &TriplePattern) -> Result<Vec<Solution>, OxirsError> {
        let mut solutions = Vec::new();

        // Get all triples from the store
        let triples = self.store.triples()?;

        for triple in triples {
            if let Some(solution) = self.match_triple_pattern(&triple, pattern) {
                solutions.push(solution);
            }
        }

        Ok(solutions)
    }

    fn match_triple_pattern(&self, triple: &Triple, pattern: &TriplePattern) -> Option<Solution> {
        let mut solution = Solution::new();

        // Match subject
        if !self.match_term_pattern(
            &Term::from_subject(triple.subject()),
            &pattern.subject,
            &mut solution,
        ) {
            return None;
        }

        // Match predicate
        if !self.match_term_pattern(
            &Term::from_predicate(triple.predicate()),
            &pattern.predicate,
            &mut solution,
        ) {
            return None;
        }

        // Match object
        if !self.match_term_pattern(
            &Term::from_object(triple.object()),
            &pattern.object,
            &mut solution,
        ) {
            return None;
        }

        Some(solution)
    }

    fn match_term_pattern(
        &self,
        term: &Term,
        pattern: &TermPattern,
        solution: &mut Solution,
    ) -> bool {
        match pattern {
            TermPattern::Variable(var) => {
                if let Some(bound_value) = solution.get(var) {
                    bound_value == term
                } else {
                    solution.bind(var.clone(), term.clone());
                    true
                }
            }
            TermPattern::NamedNode(n) => {
                matches!(term, Term::NamedNode(nn) if nn == n)
            }
            TermPattern::BlankNode(b) => {
                matches!(term, Term::BlankNode(bn) if bn == b)
            }
            TermPattern::Literal(l) => {
                matches!(term, Term::Literal(lit) if lit == l)
            }
        }
    }

    fn execute_hash_join(
        &self,
        left: &ExecutionPlan,
        right: &ExecutionPlan,
        join_vars: &[Variable],
    ) -> Result<Vec<Solution>, OxirsError> {
        let left_solutions = self.execute_plan(left)?;
        let right_solutions = self.execute_plan(right)?;

        let mut results = Vec::new();

        // Build hash table from left solutions
        let mut hash_table: HashMap<Vec<Term>, Vec<Solution>> = HashMap::new();
        for solution in left_solutions {
            let key: Vec<Term> = join_vars
                .iter()
                .filter_map(|var| solution.get(var).cloned())
                .collect();
            hash_table.entry(key).or_default().push(solution);
        }

        // Probe with right solutions
        for right_solution in right_solutions {
            let key: Vec<Term> = join_vars
                .iter()
                .filter_map(|var| right_solution.get(var).cloned())
                .collect();

            if let Some(left_solutions) = hash_table.get(&key) {
                for left_solution in left_solutions {
                    if let Some(merged) = left_solution.merge(&right_solution) {
                        results.push(merged);
                    }
                }
            }
        }

        Ok(results)
    }

    fn execute_filter(
        &self,
        input: &ExecutionPlan,
        condition: &Expression,
    ) -> Result<Vec<Solution>, OxirsError> {
        let solutions = self.execute_plan(input)?;

        Ok(solutions
            .into_iter()
            .filter(|solution| {
                self.evaluate_expression(condition, solution)
                    .unwrap_or(false)
            })
            .collect())
    }

    fn execute_project(
        &self,
        input: &ExecutionPlan,
        vars: &[Variable],
    ) -> Result<Vec<Solution>, OxirsError> {
        let solutions = self.execute_plan(input)?;

        Ok(solutions
            .into_iter()
            .map(|solution| solution.project(vars))
            .collect())
    }

    fn execute_sort(
        &self,
        input: &ExecutionPlan,
        _order_by: &[OrderExpression],
    ) -> Result<Vec<Solution>, OxirsError> {
        // Placeholder - would implement proper sorting
        self.execute_plan(input)
    }

    fn execute_limit(
        &self,
        input: &ExecutionPlan,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Solution>, OxirsError> {
        let solutions = self.execute_plan(input)?;

        Ok(solutions.into_iter().skip(offset).take(limit).collect())
    }

    fn execute_union(
        &self,
        left: &ExecutionPlan,
        right: &ExecutionPlan,
    ) -> Result<Vec<Solution>, OxirsError> {
        let mut solutions = self.execute_plan(left)?;
        solutions.extend(self.execute_plan(right)?);
        Ok(solutions)
    }

    fn execute_distinct(&self, input: &ExecutionPlan) -> Result<Vec<Solution>, OxirsError> {
        let solutions = self.execute_plan(input)?;
        let mut seen = HashSet::new();
        let mut distinct_solutions = Vec::new();

        for solution in solutions {
            if seen.insert(format!("{:?}", solution)) {
                distinct_solutions.push(solution);
            }
        }

        Ok(distinct_solutions)
    }

    fn evaluate_expression(&self, _expr: &Expression, _solution: &Solution) -> Option<bool> {
        // Placeholder - would implement full expression evaluation
        Some(true)
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self::new()
    }
}
