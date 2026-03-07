//! Conformance Test Framework
//!
//! Core types and execution engine for SPARQL 1.1 conformance testing.

pub use crate::algebra::{
    Aggregate, Algebra, BinaryOperator, Binding, Expression, GroupCondition, Literal,
    OrderCondition, PropertyPath, Term, TriplePattern, UnaryOperator, Variable,
};
pub use crate::executor::{Dataset, InMemoryDataset, QueryExecutor};
use anyhow::Result;
pub use oxirs_core::model::NamedNode;
use std::collections::HashMap;
use std::fmt;

/// Error type for conformance test failures
#[derive(Debug)]
pub struct ConformanceTestError {
    pub test_name: String,
    pub group: ConformanceGroup,
    pub expected: String,
    pub actual: String,
    pub message: String,
}

impl fmt::Display for ConformanceTestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:?}/{}] FAILED: {}\n  Expected: {}\n  Actual:   {}",
            self.group, self.test_name, self.message, self.expected, self.actual
        )
    }
}

impl std::error::Error for ConformanceTestError {}

/// SPARQL 1.1 conformance test groups based on W3C test manifest
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConformanceGroup {
    BasicPatterns,
    FilterExpressions,
    Optional,
    Union,
    Aggregates,
    GroupBy,
    Subquery,
    PropertyPaths,
    NamedGraphs,
    Update,
    Negation,
    Bind,
    Values,
    StringFunctions,
    DateFunctions,
    MathFunctions,
    TypeSystem,
    Entailment,
    Construct,
    Describe,
}

/// Expected result of a conformance test
#[derive(Debug, Clone)]
pub enum ConformanceResult {
    /// Variable bindings for SELECT queries
    SelectResults(Vec<HashMap<String, String>>),
    /// Boolean result for ASK queries
    AskResult(bool),
    /// Triples for CONSTRUCT queries: (subject, predicate, object) as string representations
    ConstructGraph(Vec<(String, String, String)>),
    /// Successful update operation
    UpdateSuccess,
    /// Test verifies error is raised
    Error,
    /// Ordered results — order matters
    OrderedSelectResults(Vec<HashMap<String, String>>),
    /// Count of results only
    ResultCount(usize),
}

/// A single W3C SPARQL 1.1 conformance test
pub struct ConformanceTest {
    pub name: &'static str,
    pub group: ConformanceGroup,
    /// SPARQL algebra to execute (pre-parsed for efficiency)
    pub algebra: Algebra,
    /// In-memory RDF dataset
    pub dataset: InMemoryDataset,
    /// Expected result
    pub expected: ConformanceResult,
}

impl ConformanceTest {
    pub fn new(
        name: &'static str,
        group: ConformanceGroup,
        algebra: Algebra,
        dataset: InMemoryDataset,
        expected: ConformanceResult,
    ) -> Self {
        Self {
            name,
            group,
            algebra,
            dataset,
            expected,
        }
    }
}

/// Runs conformance tests and reports results
pub struct ConformanceTestRunner {
    executor: QueryExecutor,
}

impl ConformanceTestRunner {
    pub fn new() -> Self {
        Self {
            executor: QueryExecutor::new(),
        }
    }

    /// Run a single conformance test
    pub fn run_test(&mut self, test: &ConformanceTest) -> Result<(), ConformanceTestError> {
        let (solution, _stats) = self
            .executor
            .execute(&test.algebra, &test.dataset)
            .map_err(|e| ConformanceTestError {
                test_name: test.name.to_string(),
                group: test.group,
                expected: format!("{:?}", test.expected),
                actual: format!("Error: {e}"),
                message: "Query execution failed".to_string(),
            })?;

        self.verify_result(test, &solution)
    }

    /// Verify query results against expected output
    fn verify_result(
        &self,
        test: &ConformanceTest,
        solution: &[Binding],
    ) -> Result<(), ConformanceTestError> {
        match &test.expected {
            ConformanceResult::ResultCount(expected_count) => {
                if solution.len() != *expected_count {
                    return Err(ConformanceTestError {
                        test_name: test.name.to_string(),
                        group: test.group,
                        expected: expected_count.to_string(),
                        actual: solution.len().to_string(),
                        message: format!(
                            "Expected {} results, got {}",
                            expected_count,
                            solution.len()
                        ),
                    });
                }
                Ok(())
            }

            ConformanceResult::AskResult(expected_bool) => {
                // ASK result is modeled as a solution with a boolean binding or empty/non-empty
                let has_results = !solution.is_empty();
                if has_results != *expected_bool {
                    return Err(ConformanceTestError {
                        test_name: test.name.to_string(),
                        group: test.group,
                        expected: expected_bool.to_string(),
                        actual: has_results.to_string(),
                        message: format!(
                            "ASK result mismatch: expected {expected_bool}, got {has_results}"
                        ),
                    });
                }
                Ok(())
            }

            ConformanceResult::SelectResults(expected_rows) => {
                self.verify_select_results(test, solution, expected_rows, false)
            }

            ConformanceResult::OrderedSelectResults(expected_rows) => {
                self.verify_select_results(test, solution, expected_rows, true)
            }

            ConformanceResult::ConstructGraph(expected_triples) => {
                // Verify triple count at minimum
                if solution.len() != expected_triples.len() {
                    return Err(ConformanceTestError {
                        test_name: test.name.to_string(),
                        group: test.group,
                        expected: format!("{} triples", expected_triples.len()),
                        actual: format!("{} solutions", solution.len()),
                        message: "CONSTRUCT result count mismatch".to_string(),
                    });
                }
                Ok(())
            }

            ConformanceResult::UpdateSuccess => Ok(()),

            ConformanceResult::Error => {
                // If we reach here, no error was raised when one was expected
                Err(ConformanceTestError {
                    test_name: test.name.to_string(),
                    group: test.group,
                    expected: "Error".to_string(),
                    actual: format!("{} results", solution.len()),
                    message: "Expected error was not raised".to_string(),
                })
            }
        }
    }

    fn verify_select_results(
        &self,
        test: &ConformanceTest,
        solution: &[Binding],
        expected_rows: &[HashMap<String, String>],
        ordered: bool,
    ) -> Result<(), ConformanceTestError> {
        if solution.len() != expected_rows.len() {
            return Err(ConformanceTestError {
                test_name: test.name.to_string(),
                group: test.group,
                expected: format!("{} rows", expected_rows.len()),
                actual: format!("{} rows", solution.len()),
                message: format!(
                    "Result count mismatch: expected {} rows, got {}",
                    expected_rows.len(),
                    solution.len()
                ),
            });
        }

        if ordered {
            // Check order-sensitive
            for (i, (actual_binding, expected_row)) in
                solution.iter().zip(expected_rows.iter()).enumerate()
            {
                if let Err(e) = self.check_row_match(test, actual_binding, expected_row) {
                    return Err(ConformanceTestError {
                        test_name: test.name.to_string(),
                        group: test.group,
                        expected: format!("row {i}: {expected_row:?}"),
                        actual: format!("row {i}: {e}"),
                        message: format!("Ordered result mismatch at row {i}"),
                    });
                }
            }
        } else {
            // Check unordered — find matching for each expected row
            let mut matched = vec![false; solution.len()];
            for expected_row in expected_rows {
                let mut found = false;
                for (idx, actual_binding) in solution.iter().enumerate() {
                    if !matched[idx]
                        && self
                            .check_row_match(test, actual_binding, expected_row)
                            .is_ok()
                    {
                        matched[idx] = true;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Err(ConformanceTestError {
                        test_name: test.name.to_string(),
                        group: test.group,
                        expected: format!("{expected_row:?}"),
                        actual: format!(
                            "{:?}",
                            solution
                                .iter()
                                .map(|b| b
                                    .iter()
                                    .map(|(k, v)| (k.to_string(), v.to_string()))
                                    .collect::<HashMap<_, _>>())
                                .collect::<Vec<_>>()
                        ),
                        message: format!("Expected row not found in results: {expected_row:?}"),
                    });
                }
            }
        }
        Ok(())
    }

    fn check_row_match(
        &self,
        _test: &ConformanceTest,
        actual: &Binding,
        expected: &HashMap<String, String>,
    ) -> Result<(), String> {
        for (var_name, expected_value) in expected {
            let var = Variable::new(var_name).map_err(|e| format!("Invalid variable: {e}"))?;
            match actual.get(&var) {
                Some(term) => {
                    let actual_str = term_to_check_string(term);
                    if actual_str != *expected_value {
                        return Err(format!(
                            "Variable ?{var_name}: expected '{expected_value}', got '{actual_str}'"
                        ));
                    }
                }
                None => {
                    // Variable unbound — treat as empty string for comparison
                    if !expected_value.is_empty() {
                        return Err(format!(
                            "Variable ?{var_name} is unbound, expected '{expected_value}'"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for ConformanceTestRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a term to its string representation for comparison
fn term_to_check_string(term: &Term) -> String {
    match term {
        Term::Iri(iri) => iri.as_str().to_string(),
        Term::Literal(lit) => lit.value.clone(),
        Term::BlankNode(id) => format!("_:{id}"),
        Term::Variable(v) => format!("?{v}"),
        Term::QuotedTriple(t) => format!("<<{} {} {}>>", t.subject, t.predicate, t.object),
        Term::PropertyPath(_) => "<path>".to_string(),
    }
}

// ============================================================
// Helper constructors for building algebra expressions
// ============================================================

/// Create an IRI term
pub fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

/// Create a named node
pub fn named_node(s: &str) -> NamedNode {
    NamedNode::new_unchecked(s)
}

/// Create a variable term
pub fn var(name: &str) -> Term {
    Term::Variable(Variable::new(name).expect("valid variable name"))
}

/// Create a variable
pub fn variable(name: &str) -> Variable {
    Variable::new(name).expect("valid variable name")
}

/// Create a plain string literal term
pub fn str_lit(s: &str) -> Term {
    Term::Literal(Literal {
        value: s.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#string",
        )),
    })
}

/// Create an integer literal term
pub fn int_lit(n: i64) -> Term {
    Term::Literal(Literal {
        value: n.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#integer",
        )),
    })
}

/// Create a decimal literal term
pub fn dec_lit(n: f64) -> Term {
    Term::Literal(Literal {
        value: n.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#decimal",
        )),
    })
}

/// Create a boolean literal term
pub fn bool_lit(b: bool) -> Term {
    Term::Literal(Literal {
        value: b.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#boolean",
        )),
    })
}

/// Create a language-tagged literal term
pub fn lang_lit(s: &str, lang: &str) -> Term {
    Term::Literal(Literal {
        value: s.to_string(),
        language: Some(lang.to_string()),
        datatype: None,
    })
}

/// Create a triple pattern
pub fn triple(s: Term, p: Term, o: Term) -> TriplePattern {
    TriplePattern::new(s, p, o)
}

/// Create a BGP algebra node
pub fn bgp(patterns: Vec<TriplePattern>) -> Algebra {
    Algebra::Bgp(patterns)
}

/// Create a join
pub fn join(left: Algebra, right: Algebra) -> Algebra {
    Algebra::Join {
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Create a left join (OPTIONAL)
pub fn left_join(left: Algebra, right: Algebra, filter: Option<Expression>) -> Algebra {
    Algebra::LeftJoin {
        left: Box::new(left),
        right: Box::new(right),
        filter,
    }
}

/// Create a union
pub fn union(left: Algebra, right: Algebra) -> Algebra {
    Algebra::Union {
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Create a filter
pub fn filter(pattern: Algebra, condition: Expression) -> Algebra {
    Algebra::Filter {
        pattern: Box::new(pattern),
        condition,
    }
}

/// Create a projection
pub fn project(pattern: Algebra, variables: Vec<Variable>) -> Algebra {
    Algebra::Project {
        pattern: Box::new(pattern),
        variables,
    }
}

/// Create a distinct
pub fn distinct(pattern: Algebra) -> Algebra {
    Algebra::Distinct {
        pattern: Box::new(pattern),
    }
}

/// Create a slice (LIMIT/OFFSET)
pub fn slice(pattern: Algebra, offset: Option<usize>, limit: Option<usize>) -> Algebra {
    Algebra::Slice {
        pattern: Box::new(pattern),
        offset,
        limit,
    }
}

/// Create an order by
pub fn order_by(pattern: Algebra, conditions: Vec<OrderCondition>) -> Algebra {
    Algebra::OrderBy {
        pattern: Box::new(pattern),
        conditions,
    }
}

/// Create an ascending order condition
pub fn asc_cond(expr: Expression) -> OrderCondition {
    OrderCondition {
        expr,
        ascending: true,
    }
}

/// Create a descending order condition
pub fn desc_cond(expr: Expression) -> OrderCondition {
    OrderCondition {
        expr,
        ascending: false,
    }
}

/// Create a group/aggregate
pub fn group(
    pattern: Algebra,
    variables: Vec<GroupCondition>,
    aggregates: Vec<(Variable, Aggregate)>,
) -> Algebra {
    Algebra::Group {
        pattern: Box::new(pattern),
        variables,
        aggregates,
    }
}

/// Create a group condition from a variable expression
pub fn group_var(var_name: &str) -> GroupCondition {
    GroupCondition {
        expr: Expression::Variable(variable(var_name)),
        alias: None,
    }
}

/// Create a VALUES clause
pub fn values(variables: Vec<Variable>, bindings: Vec<Binding>) -> Algebra {
    Algebra::Values {
        variables,
        bindings,
    }
}

/// Create a MINUS
pub fn minus(left: Algebra, right: Algebra) -> Algebra {
    Algebra::Minus {
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Create an EXTEND (BIND)
pub fn extend(pattern: Algebra, variable: Variable, expr: Expression) -> Algebra {
    Algebra::Extend {
        pattern: Box::new(pattern),
        variable,
        expr,
    }
}

/// Create an expression from a variable
pub fn expr_var(name: &str) -> Expression {
    Expression::Variable(variable(name))
}

/// Create a literal expression
pub fn expr_lit(lit: Literal) -> Expression {
    Expression::Literal(lit)
}

/// Create an IRI expression
pub fn expr_iri(s: &str) -> Expression {
    Expression::Iri(NamedNode::new_unchecked(s))
}

/// Create a binary expression
pub fn expr_binary(op: BinaryOperator, left: Expression, right: Expression) -> Expression {
    Expression::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Create an equality expression
pub fn expr_eq(left: Expression, right: Expression) -> Expression {
    expr_binary(BinaryOperator::Equal, left, right)
}

/// Create a less-than expression
pub fn expr_lt(left: Expression, right: Expression) -> Expression {
    expr_binary(BinaryOperator::Less, left, right)
}

/// Create a greater-than expression
pub fn expr_gt(left: Expression, right: Expression) -> Expression {
    expr_binary(BinaryOperator::Greater, left, right)
}

/// Create an AND expression
pub fn expr_and(left: Expression, right: Expression) -> Expression {
    expr_binary(BinaryOperator::And, left, right)
}

/// Create an OR expression
pub fn expr_or(left: Expression, right: Expression) -> Expression {
    expr_binary(BinaryOperator::Or, left, right)
}

/// Create a function call expression
pub fn expr_fn(name: &str, args: Vec<Expression>) -> Expression {
    Expression::Function {
        name: name.to_string(),
        args,
    }
}

/// Create COUNT(*) aggregate
pub fn agg_count_star() -> Aggregate {
    Aggregate::Count {
        distinct: false,
        expr: None,
    }
}

/// Create COUNT(?var)
pub fn agg_count(var_name: &str) -> Aggregate {
    Aggregate::Count {
        distinct: false,
        expr: Some(expr_var(var_name)),
    }
}

/// Create COUNT(DISTINCT ?var)
pub fn agg_count_distinct(var_name: &str) -> Aggregate {
    Aggregate::Count {
        distinct: true,
        expr: Some(expr_var(var_name)),
    }
}

/// Create SUM(?var)
pub fn agg_sum(var_name: &str) -> Aggregate {
    Aggregate::Sum {
        distinct: false,
        expr: expr_var(var_name),
    }
}

/// Create AVG(?var)
pub fn agg_avg(var_name: &str) -> Aggregate {
    Aggregate::Avg {
        distinct: false,
        expr: expr_var(var_name),
    }
}

/// Create MIN(?var)
pub fn agg_min(var_name: &str) -> Aggregate {
    Aggregate::Min {
        distinct: false,
        expr: expr_var(var_name),
    }
}

/// Create MAX(?var)
pub fn agg_max(var_name: &str) -> Aggregate {
    Aggregate::Max {
        distinct: false,
        expr: expr_var(var_name),
    }
}

/// Create GROUP_CONCAT(?var; separator=",")
pub fn agg_group_concat(var_name: &str, separator: Option<String>) -> Aggregate {
    Aggregate::GroupConcat {
        distinct: false,
        expr: expr_var(var_name),
        separator,
    }
}

/// Create a property path algebra node
pub fn property_path(subject: Term, path: PropertyPath, object: Term) -> Algebra {
    Algebra::PropertyPath {
        subject,
        path,
        object,
    }
}

/// Create a sequence property path
pub fn path_seq(left: PropertyPath, right: PropertyPath) -> PropertyPath {
    PropertyPath::Sequence(Box::new(left), Box::new(right))
}

/// Create an alternative property path
pub fn path_alt(left: PropertyPath, right: PropertyPath) -> PropertyPath {
    PropertyPath::Alternative(Box::new(left), Box::new(right))
}

/// Create a zero-or-more property path
pub fn path_star(p: PropertyPath) -> PropertyPath {
    PropertyPath::ZeroOrMore(Box::new(p))
}

/// Create a one-or-more property path
pub fn path_plus(p: PropertyPath) -> PropertyPath {
    PropertyPath::OneOrMore(Box::new(p))
}

/// Create a zero-or-one property path
pub fn path_opt(p: PropertyPath) -> PropertyPath {
    PropertyPath::ZeroOrOne(Box::new(p))
}

/// Create an inverse property path
pub fn path_inv(p: PropertyPath) -> PropertyPath {
    PropertyPath::Inverse(Box::new(p))
}

/// Create an IRI property path
pub fn path_iri(s: &str) -> PropertyPath {
    PropertyPath::Iri(NamedNode::new_unchecked(s))
}

/// Create a negated property set
pub fn path_neg(paths: Vec<PropertyPath>) -> PropertyPath {
    PropertyPath::NegatedPropertySet(paths)
}

/// Build a row for expected results
pub fn row(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Create an integer literal struct
pub fn lit_int(n: i64) -> Literal {
    Literal {
        value: n.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#integer",
        )),
    }
}

/// Create a string literal struct
pub fn lit_str(s: &str) -> Literal {
    Literal {
        value: s.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#string",
        )),
    }
}

/// Create a boolean literal struct
pub fn lit_bool(b: bool) -> Literal {
    Literal {
        value: b.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#boolean",
        )),
    }
}

/// Create a decimal literal struct
pub fn lit_dec(n: f64) -> Literal {
    Literal {
        value: n.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#decimal",
        )),
    }
}
