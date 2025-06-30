//! Query execution engine
//!
//! This module executes query plans against RDF stores.

use crate::model::*;
use crate::query::algebra::*;
use crate::query::plan::ExecutionPlan;
use crate::OxirsError;
use crate::Store;
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

    /// Returns an iterator over the variable-term bindings
    pub fn iter(&self) -> std::collections::hash_map::Iter<Variable, Term> {
        self.bindings.iter()
    }

    /// Returns an iterator over the variables in this solution
    pub fn variables(&self) -> impl Iterator<Item = &Variable> {
        self.bindings.keys()
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

    fn execute_triple_scan(
        &self,
        pattern: &crate::model::pattern::TriplePattern,
    ) -> Result<Vec<Solution>, OxirsError> {
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

    fn match_triple_pattern(
        &self,
        triple: &Triple,
        pattern: &crate::model::pattern::TriplePattern,
    ) -> Option<Solution> {
        let mut solution = Solution::new();

        // Match subject
        if let Some(ref subject_pattern) = pattern.subject {
            if !self.match_subject_pattern(triple.subject(), subject_pattern, &mut solution) {
                return None;
            }
        }

        // Match predicate
        if let Some(ref predicate_pattern) = pattern.predicate {
            if !self.match_predicate_pattern(triple.predicate(), predicate_pattern, &mut solution) {
                return None;
            }
        }

        // Match object
        if let Some(ref object_pattern) = pattern.object {
            if !self.match_object_pattern(triple.object(), object_pattern, &mut solution) {
                return None;
            }
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

    fn match_subject_pattern(
        &self,
        subject: &Subject,
        pattern: &crate::model::pattern::SubjectPattern,
        solution: &mut Solution,
    ) -> bool {
        use crate::model::pattern::SubjectPattern;
        match pattern {
            SubjectPattern::Variable(var) => {
                if let Some(bound_value) = solution.get(var) {
                    match (subject, bound_value) {
                        (Subject::NamedNode(n1), Term::NamedNode(n2)) => n1 == n2,
                        (Subject::BlankNode(b1), Term::BlankNode(b2)) => b1 == b2,
                        _ => false,
                    }
                } else {
                    solution
                        .bindings
                        .insert(var.clone(), Term::from_subject(subject));
                    true
                }
            }
            SubjectPattern::NamedNode(n) => matches!(subject, Subject::NamedNode(nn) if nn == n),
            SubjectPattern::BlankNode(b) => matches!(subject, Subject::BlankNode(bn) if bn == b),
        }
    }

    fn match_predicate_pattern(
        &self,
        predicate: &Predicate,
        pattern: &crate::model::pattern::PredicatePattern,
        solution: &mut Solution,
    ) -> bool {
        use crate::model::pattern::PredicatePattern;
        match pattern {
            PredicatePattern::Variable(var) => {
                if let Some(bound_value) = solution.get(var) {
                    match (predicate, bound_value) {
                        (Predicate::NamedNode(n1), Term::NamedNode(n2)) => n1 == n2,
                        _ => false,
                    }
                } else {
                    solution
                        .bindings
                        .insert(var.clone(), Term::from_predicate(predicate));
                    true
                }
            }
            PredicatePattern::NamedNode(n) => {
                matches!(predicate, Predicate::NamedNode(nn) if nn == n)
            }
        }
    }

    fn match_object_pattern(
        &self,
        object: &Object,
        pattern: &crate::model::pattern::ObjectPattern,
        solution: &mut Solution,
    ) -> bool {
        use crate::model::pattern::ObjectPattern;
        match pattern {
            ObjectPattern::Variable(var) => {
                if let Some(bound_value) = solution.get(var) {
                    match (object, bound_value) {
                        (Object::NamedNode(n1), Term::NamedNode(n2)) => n1 == n2,
                        (Object::BlankNode(b1), Term::BlankNode(b2)) => b1 == b2,
                        (Object::Literal(l1), Term::Literal(l2)) => l1 == l2,
                        _ => false,
                    }
                } else {
                    solution
                        .bindings
                        .insert(var.clone(), Term::from_object(object));
                    true
                }
            }
            ObjectPattern::NamedNode(n) => matches!(object, Object::NamedNode(nn) if nn == n),
            ObjectPattern::BlankNode(b) => matches!(object, Object::BlankNode(bn) if bn == b),
            ObjectPattern::Literal(l) => matches!(object, Object::Literal(lit) if lit == l),
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

    fn evaluate_expression(&self, expr: &Expression, solution: &Solution) -> Option<bool> {
        match expr {
            Expression::Variable(var) => {
                if let Some(term) = solution.get(var) {
                    // Convert term to boolean (non-empty strings and non-zero numbers are true)
                    match term {
                        Term::Literal(lit) => {
                            let value = lit.as_str();
                            match lit.datatype().as_str() {
                                "http://www.w3.org/2001/XMLSchema#boolean" => {
                                    value.parse::<bool>().ok()
                                }
                                "http://www.w3.org/2001/XMLSchema#integer"
                                | "http://www.w3.org/2001/XMLSchema#decimal"
                                | "http://www.w3.org/2001/XMLSchema#double" => {
                                    value.parse::<f64>().map(|n| n != 0.0).ok()
                                }
                                "http://www.w3.org/2001/XMLSchema#string" => {
                                    Some(!value.is_empty())
                                }
                                _ => Some(!value.is_empty()),
                            }
                        }
                        _ => Some(true), // Non-literal terms are considered true
                    }
                } else {
                    Some(false) // Unbound variables are false
                }
            }
            Expression::Literal(lit) => {
                let value = lit.as_str();
                match lit.datatype().as_str() {
                    "http://www.w3.org/2001/XMLSchema#boolean" => value.parse::<bool>().ok(),
                    "http://www.w3.org/2001/XMLSchema#integer"
                    | "http://www.w3.org/2001/XMLSchema#decimal"
                    | "http://www.w3.org/2001/XMLSchema#double" => {
                        value.parse::<f64>().map(|n| n != 0.0).ok()
                    }
                    _ => Some(!value.is_empty()),
                }
            }
            Expression::And(left, right) => {
                let left_result = self.evaluate_expression(left, solution)?;
                let right_result = self.evaluate_expression(right, solution)?;
                Some(left_result && right_result)
            }
            Expression::Or(left, right) => {
                let left_result = self.evaluate_expression(left, solution)?;
                let right_result = self.evaluate_expression(right, solution)?;
                Some(left_result || right_result)
            }
            Expression::Not(expr) => {
                let result = self.evaluate_expression(expr, solution)?;
                Some(!result)
            }
            Expression::Equal(left, right) => {
                let left_term = self.evaluate_expression_to_term(left, solution)?;
                let right_term = self.evaluate_expression_to_term(right, solution)?;
                Some(left_term == right_term)
            }
            Expression::NotEqual(left, right) => {
                let left_term = self.evaluate_expression_to_term(left, solution)?;
                let right_term = self.evaluate_expression_to_term(right, solution)?;
                Some(left_term != right_term)
            }
            Expression::Less(left, right) => {
                self.evaluate_numeric_comparison(left, right, solution, |a, b| a < b)
            }
            Expression::LessOrEqual(left, right) => {
                self.evaluate_numeric_comparison(left, right, solution, |a, b| a <= b)
            }
            Expression::Greater(left, right) => {
                self.evaluate_numeric_comparison(left, right, solution, |a, b| a > b)
            }
            Expression::GreaterOrEqual(left, right) => {
                self.evaluate_numeric_comparison(left, right, solution, |a, b| a >= b)
            }
            Expression::Bound(var) => Some(solution.get(var).is_some()),
            Expression::IsIri(expr) => {
                if let Some(term) = self.evaluate_expression_to_term(expr, solution) {
                    Some(matches!(term, Term::NamedNode(_)))
                } else {
                    Some(false)
                }
            }
            Expression::IsBlank(expr) => {
                if let Some(term) = self.evaluate_expression_to_term(expr, solution) {
                    Some(matches!(term, Term::BlankNode(_)))
                } else {
                    Some(false)
                }
            }
            Expression::IsLiteral(expr) => {
                if let Some(term) = self.evaluate_expression_to_term(expr, solution) {
                    Some(matches!(term, Term::Literal(_)))
                } else {
                    Some(false)
                }
            }
            Expression::IsNumeric(expr) => {
                if let Some(Term::Literal(lit)) = self.evaluate_expression_to_term(expr, solution) {
                    let datatype_str = lit.datatype().as_str().to_string();
                    Some(matches!(
                        datatype_str.as_str(),
                        "http://www.w3.org/2001/XMLSchema#integer"
                            | "http://www.w3.org/2001/XMLSchema#decimal"
                            | "http://www.w3.org/2001/XMLSchema#double"
                            | "http://www.w3.org/2001/XMLSchema#float"
                    ))
                } else {
                    Some(false)
                }
            }
            Expression::Str(expr) => {
                // STR() always succeeds, so it's always "true" for filtering purposes
                Some(self.evaluate_expression_to_term(expr, solution).is_some())
            }
            Expression::Regex(text_expr, pattern_expr, flags_expr) => {
                let text = self.evaluate_expression_to_string(text_expr, solution)?;
                let pattern = self.evaluate_expression_to_string(pattern_expr, solution)?;

                let flags = if let Some(flags_expr) = flags_expr {
                    self.evaluate_expression_to_string(flags_expr, solution)
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                // Basic regex implementation (would need full regex crate for production)
                if flags.is_empty() {
                    Some(text.contains(&pattern))
                } else {
                    // For now, just do case-insensitive matching if 'i' flag is present
                    if flags.contains('i') {
                        Some(text.to_lowercase().contains(&pattern.to_lowercase()))
                    } else {
                        Some(text.contains(&pattern))
                    }
                }
            }
            _ => {
                // For unsupported expressions, default to true
                // This is a simplified implementation
                Some(true)
            }
        }
    }

    /// Evaluate an expression to a term value
    fn evaluate_expression_to_term(&self, expr: &Expression, solution: &Solution) -> Option<Term> {
        match expr {
            Expression::Variable(var) => solution.get(var).cloned(),
            Expression::Term(term) => Some(term.clone()),
            Expression::FunctionCall(Function::Str, args) => {
                if let Some(arg) = args.first() {
                    if let Some(term) = self.evaluate_expression_to_term(arg, solution) {
                        match term {
                            Term::NamedNode(n) => Some(Term::Literal(Literal::new(n.as_str()))),
                            Term::Literal(l) => Some(Term::Literal(Literal::new(l.as_str()))),
                            Term::BlankNode(b) => Some(Term::Literal(Literal::new(b.as_str()))),
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None, // Other expressions don't directly evaluate to terms
        }
    }

    /// Evaluate an expression to a string value
    fn evaluate_expression_to_string(
        &self,
        expr: &Expression,
        solution: &Solution,
    ) -> Option<String> {
        if let Some(term) = self.evaluate_expression_to_term(expr, solution) {
            match term {
                Term::NamedNode(n) => Some(n.as_str().to_string()),
                Term::Literal(l) => Some(l.as_str().to_string()),
                Term::BlankNode(b) => Some(b.as_str().to_string()),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Evaluate a numeric comparison
    fn evaluate_numeric_comparison<F>(
        &self,
        left: &Expression,
        right: &Expression,
        solution: &Solution,
        comparator: F,
    ) -> Option<bool>
    where
        F: Fn(f64, f64) -> bool,
    {
        let left_val = self.evaluate_expression_to_numeric(left, solution)?;
        let right_val = self.evaluate_expression_to_numeric(right, solution)?;
        Some(comparator(left_val, right_val))
    }

    /// Evaluate an expression to a numeric value
    fn evaluate_expression_to_numeric(
        &self,
        expr: &Expression,
        solution: &Solution,
    ) -> Option<f64> {
        if let Some(Term::Literal(lit)) = self.evaluate_expression_to_term(expr, solution) {
            let value = lit.as_str();
            match lit.datatype().as_str() {
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => value.parse::<f64>().ok(),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self::new()
    }
}
