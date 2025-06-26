//! SPARQL-star built-in functions for working with quoted triples.
//!
//! This module provides the standard SPARQL-star functions defined in the
//! specification for constructing and deconstructing quoted triples.

use crate::model::{StarTerm, StarTriple};
use crate::{StarError, StarResult};

/// SPARQL-star built-in functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StarFunction {
    /// TRIPLE(s, p, o) - constructs a quoted triple
    Triple,
    /// SUBJECT(t) - extracts the subject from a quoted triple
    Subject,
    /// PREDICATE(t) - extracts the predicate from a quoted triple
    Predicate,
    /// OBJECT(t) - extracts the object from a quoted triple
    Object,
    /// isTRIPLE(t) - tests if a term is a quoted triple
    IsTriple,
}

impl StarFunction {
    /// Get the function name as a string
    pub fn name(&self) -> &'static str {
        match self {
            StarFunction::Triple => "TRIPLE",
            StarFunction::Subject => "SUBJECT",
            StarFunction::Predicate => "PREDICATE",
            StarFunction::Object => "OBJECT",
            StarFunction::IsTriple => "isTRIPLE",
        }
    }

    /// Get the expected number of arguments for this function
    pub fn arity(&self) -> usize {
        match self {
            StarFunction::Triple => 3,
            StarFunction::Subject => 1,
            StarFunction::Predicate => 1,
            StarFunction::Object => 1,
            StarFunction::IsTriple => 1,
        }
    }

    /// Parse a function name into a StarFunction
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "TRIPLE" => Some(StarFunction::Triple),
            "SUBJECT" => Some(StarFunction::Subject),
            "PREDICATE" => Some(StarFunction::Predicate),
            "OBJECT" => Some(StarFunction::Object),
            "ISTRIPLE" => Some(StarFunction::IsTriple),
            _ => None,
        }
    }
}

/// SPARQL-star function evaluator
pub struct FunctionEvaluator;

impl FunctionEvaluator {
    /// Evaluate a SPARQL-star function with the given arguments
    pub fn evaluate(function: StarFunction, args: &[StarTerm]) -> StarResult<StarTerm> {
        // Check argument count
        let expected_arity = function.arity();
        if args.len() != expected_arity {
            return Err(StarError::QueryError(format!(
                "Function {} expects {} arguments, got {}",
                function.name(),
                expected_arity,
                args.len()
            )));
        }

        match function {
            StarFunction::Triple => Self::evaluate_triple(&args[0], &args[1], &args[2]),
            StarFunction::Subject => Self::evaluate_subject(&args[0]),
            StarFunction::Predicate => Self::evaluate_predicate(&args[0]),
            StarFunction::Object => Self::evaluate_object(&args[0]),
            StarFunction::IsTriple => Self::evaluate_is_triple(&args[0]),
        }
    }

    /// TRIPLE(s, p, o) - constructs a quoted triple from subject, predicate, object
    fn evaluate_triple(subject: &StarTerm, predicate: &StarTerm, object: &StarTerm) -> StarResult<StarTerm> {
        // Validate that the arguments can form a valid triple
        if !subject.can_be_subject() {
            return Err(StarError::QueryError(format!(
                "Invalid subject for TRIPLE function: {:?}",
                subject
            )));
        }

        if !predicate.can_be_predicate() {
            return Err(StarError::QueryError(format!(
                "Invalid predicate for TRIPLE function: {:?}",
                predicate
            )));
        }

        if !object.can_be_object() {
            return Err(StarError::QueryError(format!(
                "Invalid object for TRIPLE function: {:?}",
                object
            )));
        }

        // Create the quoted triple
        let triple = StarTriple::new(subject.clone(), predicate.clone(), object.clone());
        Ok(StarTerm::quoted_triple(triple))
    }

    /// SUBJECT(t) - extracts the subject from a quoted triple
    fn evaluate_subject(term: &StarTerm) -> StarResult<StarTerm> {
        match term {
            StarTerm::QuotedTriple(triple) => Ok(triple.subject.clone()),
            _ => Err(StarError::QueryError(format!(
                "SUBJECT function expects a quoted triple, got: {:?}",
                term
            ))),
        }
    }

    /// PREDICATE(t) - extracts the predicate from a quoted triple
    fn evaluate_predicate(term: &StarTerm) -> StarResult<StarTerm> {
        match term {
            StarTerm::QuotedTriple(triple) => Ok(triple.predicate.clone()),
            _ => Err(StarError::QueryError(format!(
                "PREDICATE function expects a quoted triple, got: {:?}",
                term
            ))),
        }
    }

    /// OBJECT(t) - extracts the object from a quoted triple
    fn evaluate_object(term: &StarTerm) -> StarResult<StarTerm> {
        match term {
            StarTerm::QuotedTriple(triple) => Ok(triple.object.clone()),
            _ => Err(StarError::QueryError(format!(
                "OBJECT function expects a quoted triple, got: {:?}",
                term
            ))),
        }
    }

    /// isTRIPLE(t) - tests if a term is a quoted triple
    fn evaluate_is_triple(term: &StarTerm) -> StarResult<StarTerm> {
        let result = term.is_quoted_triple();
        
        // Return a boolean literal (xsd:boolean)
        let value = if result { "true" } else { "false" };
        StarTerm::literal_with_datatype(value, "http://www.w3.org/2001/XMLSchema#boolean")
    }
}

/// Expression type for SPARQL-star expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Constant term
    Term(StarTerm),
    /// Variable reference
    Variable(String),
    /// Function call
    FunctionCall {
        function: StarFunction,
        args: Vec<Expression>,
    },
    /// Binary operation (e.g., equality, comparison)
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    /// Unary operation (e.g., negation)
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
}

/// Binary operators for SPARQL expressions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOperator {
    /// Equality (=)
    Equal,
    /// Inequality (!=)
    NotEqual,
    /// Less than (<)
    LessThan,
    /// Less than or equal (<=)
    LessThanOrEqual,
    /// Greater than (>)
    GreaterThan,
    /// Greater than or equal (>=)
    GreaterThanOrEqual,
    /// Logical AND (&&)
    And,
    /// Logical OR (||)
    Or,
}

/// Unary operators for SPARQL expressions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOperator {
    /// Logical NOT (!)
    Not,
    /// Arithmetic negation (-)
    Minus,
}

/// Expression evaluator for SPARQL-star expressions
pub struct ExpressionEvaluator;

impl ExpressionEvaluator {
    /// Evaluate an expression with the given variable bindings
    pub fn evaluate(
        expr: &Expression,
        bindings: &std::collections::HashMap<String, StarTerm>,
    ) -> StarResult<StarTerm> {
        match expr {
            Expression::Term(term) => Ok(term.clone()),
            Expression::Variable(var) => {
                bindings.get(var).cloned().ok_or_else(|| {
                    StarError::QueryError(format!("Unbound variable: {}", var))
                })
            }
            Expression::FunctionCall { function, args } => {
                // Evaluate arguments first
                let evaluated_args: Result<Vec<_>, _> = args
                    .iter()
                    .map(|arg| Self::evaluate(arg, bindings))
                    .collect();
                
                let evaluated_args = evaluated_args?;
                FunctionEvaluator::evaluate(*function, &evaluated_args)
            }
            Expression::BinaryOp { op, left, right } => {
                let left_val = Self::evaluate(left, bindings)?;
                let right_val = Self::evaluate(right, bindings)?;
                Self::evaluate_binary_op(*op, &left_val, &right_val)
            }
            Expression::UnaryOp { op, expr } => {
                let val = Self::evaluate(expr, bindings)?;
                Self::evaluate_unary_op(*op, &val)
            }
        }
    }

    /// Evaluate a binary operation
    fn evaluate_binary_op(
        op: BinaryOperator,
        left: &StarTerm,
        right: &StarTerm,
    ) -> StarResult<StarTerm> {
        match op {
            BinaryOperator::Equal => {
                let result = left == right;
                StarTerm::literal_with_datatype(
                    if result { "true" } else { "false" },
                    "http://www.w3.org/2001/XMLSchema#boolean",
                )
            }
            BinaryOperator::NotEqual => {
                let result = left != right;
                StarTerm::literal_with_datatype(
                    if result { "true" } else { "false" },
                    "http://www.w3.org/2001/XMLSchema#boolean",
                )
            }
            // Other operators would require type checking and numeric comparisons
            _ => Err(StarError::QueryError(format!(
                "Binary operator {:?} not yet implemented",
                op
            ))),
        }
    }

    /// Evaluate a unary operation
    fn evaluate_unary_op(op: UnaryOperator, term: &StarTerm) -> StarResult<StarTerm> {
        match op {
            UnaryOperator::Not => {
                // Expects a boolean literal
                if let Some(literal) = term.as_literal() {
                    if let Some(datatype) = &literal.datatype {
                        if datatype.iri == "http://www.w3.org/2001/XMLSchema#boolean" {
                            let value = literal.value.as_str();
                            let negated = match value {
                                "true" => "false",
                                "false" => "true",
                                _ => return Err(StarError::QueryError(
                                    "Invalid boolean value".to_string()
                                )),
                            };
                            return StarTerm::literal_with_datatype(
                                negated,
                                "http://www.w3.org/2001/XMLSchema#boolean",
                            );
                        }
                    }
                }
                Err(StarError::QueryError(
                    "NOT operator expects a boolean literal".to_string()
                ))
            }
            UnaryOperator::Minus => Err(StarError::QueryError(
                "Arithmetic operations not yet implemented".to_string()
            )),
        }
    }
}

/// Helper functions for constructing expressions
impl Expression {
    /// Create a TRIPLE function call expression
    pub fn triple(subject: Expression, predicate: Expression, object: Expression) -> Self {
        Expression::FunctionCall {
            function: StarFunction::Triple,
            args: vec![subject, predicate, object],
        }
    }

    /// Create a SUBJECT function call expression
    pub fn subject(expr: Expression) -> Self {
        Expression::FunctionCall {
            function: StarFunction::Subject,
            args: vec![expr],
        }
    }

    /// Create a PREDICATE function call expression
    pub fn predicate(expr: Expression) -> Self {
        Expression::FunctionCall {
            function: StarFunction::Predicate,
            args: vec![expr],
        }
    }

    /// Create an OBJECT function call expression
    pub fn object(expr: Expression) -> Self {
        Expression::FunctionCall {
            function: StarFunction::Object,
            args: vec![expr],
        }
    }

    /// Create an isTRIPLE function call expression
    pub fn is_triple(expr: Expression) -> Self {
        Expression::FunctionCall {
            function: StarFunction::IsTriple,
            args: vec![expr],
        }
    }

    /// Create an equality expression
    pub fn equal(left: Expression, right: Expression) -> Self {
        Expression::BinaryOp {
            op: BinaryOperator::Equal,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a variable expression
    pub fn var(name: &str) -> Self {
        Expression::Variable(name.to_string())
    }

    /// Create a term expression
    pub fn term(term: StarTerm) -> Self {
        Expression::Term(term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_triple_function() {
        let subject = StarTerm::iri("http://example.org/alice").unwrap();
        let predicate = StarTerm::iri("http://example.org/age").unwrap();
        let object = StarTerm::literal("25").unwrap();

        let result = FunctionEvaluator::evaluate(
            StarFunction::Triple,
            &[subject.clone(), predicate.clone(), object.clone()],
        ).unwrap();

        assert!(result.is_quoted_triple());
        if let StarTerm::QuotedTriple(triple) = result {
            assert_eq!(triple.subject, subject);
            assert_eq!(triple.predicate, predicate);
            assert_eq!(triple.object, object);
        }
    }

    #[test]
    fn test_subject_function() {
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let quoted = StarTerm::quoted_triple(triple);

        let result = FunctionEvaluator::evaluate(
            StarFunction::Subject,
            &[quoted],
        ).unwrap();

        assert_eq!(result, StarTerm::iri("http://example.org/alice").unwrap());
    }

    #[test]
    fn test_predicate_function() {
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let quoted = StarTerm::quoted_triple(triple);

        let result = FunctionEvaluator::evaluate(
            StarFunction::Predicate,
            &[quoted],
        ).unwrap();

        assert_eq!(result, StarTerm::iri("http://example.org/age").unwrap());
    }

    #[test]
    fn test_object_function() {
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let quoted = StarTerm::quoted_triple(triple);

        let result = FunctionEvaluator::evaluate(
            StarFunction::Object,
            &[quoted],
        ).unwrap();

        assert_eq!(result, StarTerm::literal("25").unwrap());
    }

    #[test]
    fn test_is_triple_function() {
        // Test with a quoted triple
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let quoted = StarTerm::quoted_triple(triple);

        let result = FunctionEvaluator::evaluate(
            StarFunction::IsTriple,
            &[quoted],
        ).unwrap();

        assert_eq!(
            result,
            StarTerm::literal_with_datatype("true", "http://www.w3.org/2001/XMLSchema#boolean").unwrap()
        );

        // Test with a non-quoted term
        let iri = StarTerm::iri("http://example.org/alice").unwrap();
        let result = FunctionEvaluator::evaluate(
            StarFunction::IsTriple,
            &[iri],
        ).unwrap();

        assert_eq!(
            result,
            StarTerm::literal_with_datatype("false", "http://www.w3.org/2001/XMLSchema#boolean").unwrap()
        );
    }

    #[test]
    fn test_expression_evaluation() {
        let mut bindings = HashMap::new();
        bindings.insert(
            "x".to_string(),
            StarTerm::iri("http://example.org/alice").unwrap(),
        );
        bindings.insert(
            "p".to_string(),
            StarTerm::iri("http://example.org/age").unwrap(),
        );
        bindings.insert(
            "o".to_string(),
            StarTerm::literal("25").unwrap(),
        );

        // Test TRIPLE(?x, ?p, ?o)
        let expr = Expression::triple(
            Expression::var("x"),
            Expression::var("p"),
            Expression::var("o"),
        );

        let result = ExpressionEvaluator::evaluate(&expr, &bindings).unwrap();
        assert!(result.is_quoted_triple());

        // Test SUBJECT(TRIPLE(?x, ?p, ?o))
        let expr = Expression::subject(Expression::triple(
            Expression::var("x"),
            Expression::var("p"),
            Expression::var("o"),
        ));

        let result = ExpressionEvaluator::evaluate(&expr, &bindings).unwrap();
        assert_eq!(result, StarTerm::iri("http://example.org/alice").unwrap());
    }

    #[test]
    fn test_nested_function_calls() {
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );
        let quoted1 = StarTerm::quoted_triple(triple1);

        let triple2 = StarTriple::new(
            quoted1.clone(),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        let quoted2 = StarTerm::quoted_triple(triple2);

        // Test SUBJECT(SUBJECT(<<...>>))
        let inner_subject = FunctionEvaluator::evaluate(
            StarFunction::Subject,
            &[quoted2.clone()],
        ).unwrap();

        let outer_subject = FunctionEvaluator::evaluate(
            StarFunction::Subject,
            &[inner_subject],
        ).unwrap();

        assert_eq!(outer_subject, StarTerm::iri("http://example.org/alice").unwrap());
    }

    #[test]
    fn test_error_cases() {
        // Wrong number of arguments
        let result = FunctionEvaluator::evaluate(
            StarFunction::Triple,
            &[StarTerm::iri("http://example.org/alice").unwrap()],
        );
        assert!(result.is_err());

        // Invalid subject for TRIPLE
        let result = FunctionEvaluator::evaluate(
            StarFunction::Triple,
            &[
                StarTerm::literal("invalid_subject").unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::literal("o").unwrap(),
            ],
        );
        assert!(result.is_err());

        // Non-triple argument to SUBJECT
        let result = FunctionEvaluator::evaluate(
            StarFunction::Subject,
            &[StarTerm::iri("http://example.org/notATriple").unwrap()],
        );
        assert!(result.is_err());
    }
}