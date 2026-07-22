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
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, Variable, Term> {
        self.bindings.iter()
    }

    /// Returns an iterator over the variables in this solution
    pub fn variables(&self) -> impl Iterator<Item = &Variable> {
        self.bindings.keys()
    }

    /// Build a deterministic, order-independent key for this solution.
    ///
    /// The bindings live in a `HashMap` whose iteration order is not stable
    /// between instances, so any key derived from raw iteration order (such as
    /// the `Debug` string) is unreliable for equality/deduplication. Sorting the
    /// `(variable, term)` pairs by their canonical string form yields a key that
    /// is identical for any two solutions with identical bindings.
    pub fn canonical_key(&self) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = self
            .bindings
            .iter()
            .map(|(var, term)| (var.name().to_string(), term.to_string()))
            .collect();
        pairs.sort();
        pairs
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
    store: &'a dyn Store,
}

impl<'a> QueryExecutor<'a> {
    /// Creates a new query executor
    pub fn new(store: &'a dyn Store) -> Self {
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

        // SPARQL default-graph semantics: a BGP outside a GRAPH clause must
        // match ONLY the active (default) graph, never the union of the default
        // graph and every named graph. Scanning `store.triples()` (which unions
        // all graphs) would incorrectly surface triples that live exclusively in
        // named graphs. Restrict the scan to the default graph.
        let quads = self.store.default_graph_quads()?;

        for quad in quads {
            let triple = Triple::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
            );
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

    #[allow(dead_code)]
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
            TermPattern::QuotedTriple(_) => {
                // Fine-grained RDF-star quoted-triple matching is not implemented
                // in this helper. Rather than crashing the query thread, treat
                // the pattern as non-matching so an unsupported pattern degrades
                // gracefully instead of panicking.
                false
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
            // A quoted-triple pattern matches any quoted-triple subject; variable binding
            // refinement for the inner triple is handled at a higher level.
            SubjectPattern::QuotedTriple(_) => matches!(subject, Subject::QuotedTriple(_)),
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
            // A quoted-triple pattern matches any quoted-triple object.
            ObjectPattern::QuotedTriple(_) => matches!(object, Object::QuotedTriple(_)),
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
        order_by: &[OrderExpression],
    ) -> Result<Vec<Solution>, OxirsError> {
        let mut solutions = self.execute_plan(input)?;

        // Stable sort so that equal keys preserve their relative input order.
        solutions.sort_by(|a, b| {
            for order in order_by {
                let (expr, descending) = match order {
                    OrderExpression::Asc(e) => (e, false),
                    OrderExpression::Desc(e) => (e, true),
                };
                let ta = self.evaluate_expression_to_term(expr, a);
                let tb = self.evaluate_expression_to_term(expr, b);
                let mut ord = Self::order_compare(ta.as_ref(), tb.as_ref());
                if descending {
                    ord = ord.reverse();
                }
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });

        Ok(solutions)
    }

    /// Total ordering used by `ORDER BY`, following the SPARQL term ordering:
    /// unbound values sort first, then blank nodes, IRIs, and literals; within a
    /// kind, values are compared by their typed value (falling back to lexical
    /// order for otherwise-incomparable literals so the sort stays total and
    /// deterministic).
    fn order_compare(a: Option<&Term>, b: Option<&Term>) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(a), Some(b)) => {
                if let Some(ord) = Self::compare_terms(a, b) {
                    return ord;
                }
                Self::term_kind_rank(a)
                    .cmp(&Self::term_kind_rank(b))
                    .then_with(|| a.to_string().cmp(&b.to_string()))
            }
        }
    }

    /// Rank of an RDF term kind for the total ORDER BY ordering.
    fn term_kind_rank(term: &Term) -> u8 {
        match term {
            Term::BlankNode(_) => 0,
            Term::NamedNode(_) => 1,
            Term::Literal(_) => 2,
            _ => 3,
        }
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
            // Build a canonical, order-independent key. `Solution` wraps a
            // `HashMap`, whose `Debug` iteration order varies per instance (the
            // hasher is seeded per map), so hashing `format!("{solution:?}")`
            // could give two identical binding sets different keys and fail to
            // deduplicate them. Sorting the (variable, term) pairs makes the key
            // deterministic.
            if seen.insert(solution.canonical_key()) {
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
            Expression::Less(left, right) => self
                .compare_terms_expr(left, right, solution)
                .map(|ord| ord == std::cmp::Ordering::Less),
            Expression::LessOrEqual(left, right) => self
                .compare_terms_expr(left, right, solution)
                .map(|ord| ord != std::cmp::Ordering::Greater),
            Expression::Greater(left, right) => self
                .compare_terms_expr(left, right, solution)
                .map(|ord| ord == std::cmp::Ordering::Greater),
            Expression::GreaterOrEqual(left, right) => self
                .compare_terms_expr(left, right, solution)
                .map(|ord| ord != std::cmp::Ordering::Less),
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
    #[allow(clippy::only_used_in_recursion)]
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

    /// Evaluate the ordered comparison of two expressions per the SPARQL
    /// operator-mapping rules, returning the [`Ordering`](std::cmp::Ordering)
    /// of the two operand values.
    ///
    /// Supports numeric (`xsd:integer`/`decimal`/`double`/`float`), `xsd:string`
    /// (Unicode codepoint order), `xsd:boolean` (`false` < `true`), and
    /// `xsd:date`/`xsd:dateTime` (timezone-aware temporal order) operands.
    ///
    /// Returns `None` for genuinely incomparable operand pairs (a SPARQL type
    /// error). Per SPARQL semantics a type error in a FILTER excludes the
    /// solution, so `None` is mapped to a dropped row by the caller — never a
    /// silently over-broad result.
    fn compare_terms_expr(
        &self,
        left: &Expression,
        right: &Expression,
        solution: &Solution,
    ) -> Option<std::cmp::Ordering> {
        let left_term = self.evaluate_expression_to_term(left, solution)?;
        let right_term = self.evaluate_expression_to_term(right, solution)?;
        Self::compare_terms(&left_term, &right_term)
    }

    /// Compare two RDF terms per the SPARQL/XPath operator mapping.
    ///
    /// Returns `None` when the operands are of incomparable kinds (a type
    /// error under SPARQL semantics).
    fn compare_terms(left: &Term, right: &Term) -> Option<std::cmp::Ordering> {
        let (l, r) = match (left, right) {
            (Term::Literal(l), Term::Literal(r)) => (l, r),
            // Ordered comparison is only defined between literals.
            _ => return None,
        };

        let l_dt = l.datatype().as_str().to_string();
        let r_dt = r.datatype().as_str().to_string();

        // Numeric comparison (mixed numeric datatypes promote to f64).
        if let (Some(a), Some(b)) = (
            Self::numeric_value(&l_dt, l.as_str()),
            Self::numeric_value(&r_dt, r.as_str()),
        ) {
            return a.partial_cmp(&b);
        }

        // Both operands must share the same datatype family for the remaining
        // comparisons.
        if l_dt != r_dt {
            return None;
        }

        match l_dt.as_str() {
            "http://www.w3.org/2001/XMLSchema#string" => Some(l.as_str().cmp(r.as_str())),
            "http://www.w3.org/2001/XMLSchema#boolean" => {
                let a = Self::boolean_value(l.as_str())?;
                let b = Self::boolean_value(r.as_str())?;
                Some(a.cmp(&b))
            }
            "http://www.w3.org/2001/XMLSchema#dateTime" => {
                use std::str::FromStr;
                let a = oxsdatatypes::DateTime::from_str(l.as_str()).ok()?;
                let b = oxsdatatypes::DateTime::from_str(r.as_str()).ok()?;
                a.partial_cmp(&b)
            }
            "http://www.w3.org/2001/XMLSchema#date" => {
                use std::str::FromStr;
                let a = oxsdatatypes::Date::from_str(l.as_str()).ok()?;
                let b = oxsdatatypes::Date::from_str(r.as_str()).ok()?;
                a.partial_cmp(&b)
            }
            _ => None,
        }
    }

    /// Parse a numeric literal value into an `f64`, or `None` if the datatype is
    /// not a recognised XSD numeric type.
    fn numeric_value(datatype: &str, value: &str) -> Option<f64> {
        match datatype {
            "http://www.w3.org/2001/XMLSchema#integer"
            | "http://www.w3.org/2001/XMLSchema#decimal"
            | "http://www.w3.org/2001/XMLSchema#double"
            | "http://www.w3.org/2001/XMLSchema#float" => value.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Parse an `xsd:boolean` lexical form (`true`/`1`, `false`/`0`).
    fn boolean_value(value: &str) -> Option<bool> {
        match value {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject};
    use crate::query::{QueryEngine, QueryResult};
    use crate::rdf_store::RdfStore;
    use crate::Store;

    fn iri(s: &str) -> NamedNode {
        NamedNode::new(s).expect("valid iri")
    }

    fn count_bindings(result: &QueryResult) -> usize {
        match result {
            QueryResult::Select { bindings, .. } => bindings.len(),
            _ => panic!("expected SELECT result"),
        }
    }

    /// P1: a BGP outside a GRAPH clause must match only the default graph, not
    /// the union of the default graph and every named graph.
    #[test]
    fn regression_bgp_scans_default_graph_only() {
        let store = RdfStore::new().expect("store");
        // Default-graph triple.
        store
            .insert_quad(Quad::new(
                Subject::NamedNode(iri("http://example.org/s1")),
                Predicate::NamedNode(iri("http://example.org/p")),
                Object::NamedNode(iri("http://example.org/o1")),
                GraphName::DefaultGraph,
            ))
            .expect("insert default");
        // Named-graph-only triple: must NOT be visible to a default-graph BGP.
        store
            .insert_quad(Quad::new(
                Subject::NamedNode(iri("http://example.org/s2")),
                Predicate::NamedNode(iri("http://example.org/p")),
                Object::NamedNode(iri("http://example.org/o2")),
                GraphName::NamedNode(iri("http://example.org/g")),
            ))
            .expect("insert named");

        let engine = QueryEngine::new();
        let result = engine
            .query("SELECT * WHERE { ?s ?p ?o . }", &store)
            .expect("query ok");
        assert_eq!(
            count_bindings(&result),
            1,
            "only the default-graph triple must match a default-graph BGP"
        );
    }

    /// P2: DISTINCT must deduplicate multi-variable solutions reliably.
    #[test]
    fn regression_distinct_multi_variable_dedup() {
        let store = RdfStore::new().expect("store");
        let p = iri("http://example.org/p");
        let o = iri("http://example.org/o");
        // Two distinct subjects sharing identical (p, o) bindings.
        for s in ["http://example.org/s1", "http://example.org/s2"] {
            store
                .insert_quad(Quad::new(
                    Subject::NamedNode(iri(s)),
                    Predicate::NamedNode(p.clone()),
                    Object::NamedNode(o.clone()),
                    GraphName::DefaultGraph,
                ))
                .expect("insert");
        }

        let engine = QueryEngine::new();
        let result = engine
            .query("SELECT DISTINCT ?p ?o WHERE { ?s ?p ?o . }", &store)
            .expect("query ok");
        assert_eq!(
            count_bindings(&result),
            1,
            "identical (p, o) bindings must collapse to a single DISTINCT row"
        );
    }

    /// P2: FILTER string comparison must perform a real ordered comparison, not
    /// drop every row.
    #[test]
    fn regression_filter_string_comparison() {
        let store = RdfStore::new().expect("store");
        let p = iri("http://example.org/name");
        for (s, name) in [
            ("http://example.org/a", "apple"),
            ("http://example.org/z", "zebra"),
        ] {
            store
                .insert_quad(Quad::new(
                    Subject::NamedNode(iri(s)),
                    Predicate::NamedNode(p.clone()),
                    Object::Literal(Literal::new(name)),
                    GraphName::DefaultGraph,
                ))
                .expect("insert");
        }

        let engine = QueryEngine::new();
        let result = engine
            .query(
                "SELECT ?name WHERE { ?s ?p ?name . FILTER(?name > \"m\") }",
                &store,
            )
            .expect("query ok");
        assert_eq!(
            count_bindings(&result),
            1,
            "only 'zebra' is greater than 'm'"
        );
    }

    /// P2: FILTER date comparison must compare temporal values, not drop rows.
    #[test]
    fn regression_filter_date_comparison() {
        let store = RdfStore::new().expect("store");
        let p = iri("http://example.org/born");
        let date_dt = crate::vocab::xsd::DATE.clone();
        for (s, d) in [
            ("http://example.org/old", "1990-01-01"),
            ("http://example.org/new", "2020-01-01"),
        ] {
            store
                .insert_quad(Quad::new(
                    Subject::NamedNode(iri(s)),
                    Predicate::NamedNode(p.clone()),
                    Object::Literal(Literal::new_typed(d, date_dt.clone())),
                    GraphName::DefaultGraph,
                ))
                .expect("insert");
        }

        let engine = QueryEngine::new();
        let result = engine
            .query(
                "SELECT ?d WHERE { ?s ?p ?d . FILTER(?d < \"2000-01-01\"^^<http://www.w3.org/2001/XMLSchema#date>) }",
                &store,
            )
            .expect("query ok");
        assert_eq!(
            count_bindings(&result),
            1,
            "only the 1990 date is before 2000"
        );
    }

    /// P1: ORDER BY must actually order the results.
    #[test]
    fn regression_order_by_sorts_results() {
        let store = RdfStore::new().expect("store");
        let p = iri("http://example.org/n");
        let int_dt = crate::vocab::xsd::INTEGER.clone();
        for (s, n) in [
            ("http://example.org/b", "3"),
            ("http://example.org/a", "1"),
            ("http://example.org/c", "2"),
        ] {
            store
                .insert_quad(Quad::new(
                    Subject::NamedNode(iri(s)),
                    Predicate::NamedNode(p.clone()),
                    Object::Literal(Literal::new_typed(n, int_dt.clone())),
                    GraphName::DefaultGraph,
                ))
                .expect("insert");
        }

        let engine = QueryEngine::new();
        let result = engine
            .query("SELECT ?n WHERE { ?s ?p ?n . } ORDER BY ?n", &store)
            .expect("query ok");
        let QueryResult::Select { bindings, .. } = result else {
            panic!("expected SELECT");
        };
        let values: Vec<String> = bindings
            .iter()
            .filter_map(|b| b.get("n"))
            .map(|t| t.to_string())
            .collect();
        let sorted_positions: Vec<&String> = values.iter().collect();
        // Values must be in ascending numeric order: 1, 2, 3.
        assert_eq!(values.len(), 3);
        assert!(
            sorted_positions[0].contains('1')
                && sorted_positions[1].contains('2')
                && sorted_positions[2].contains('3'),
            "ORDER BY ?n should yield 1,2,3 — got {values:?}"
        );
    }

    /// P1: LIMIT/OFFSET must bound the result set.
    #[test]
    fn regression_limit_offset_applied() {
        let store = RdfStore::new().expect("store");
        let p = iri("http://example.org/n");
        let int_dt = crate::vocab::xsd::INTEGER.clone();
        for n in 0..5 {
            store
                .insert_quad(Quad::new(
                    Subject::NamedNode(iri(&format!("http://example.org/s{n}"))),
                    Predicate::NamedNode(p.clone()),
                    Object::Literal(Literal::new_typed(n.to_string(), int_dt.clone())),
                    GraphName::DefaultGraph,
                ))
                .expect("insert");
        }

        let engine = QueryEngine::new();
        let result = engine
            .query(
                "SELECT ?n WHERE { ?s ?p ?n . } ORDER BY ?n LIMIT 2 OFFSET 1",
                &store,
            )
            .expect("query ok");
        assert_eq!(
            count_bindings(&result),
            2,
            "LIMIT 2 must cap the result set"
        );
    }
}
