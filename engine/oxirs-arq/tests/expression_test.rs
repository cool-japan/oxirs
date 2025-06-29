//! Tests for the expression evaluation system

use oxirs_arq::algebra::{BinaryOperator, Expression, Literal, UnaryOperator, Variable};
use oxirs_arq::expression::ExpressionEvaluator;
use oxirs_arq::extensions::ExtensionRegistry;
use oxirs_arq::term::{xsd, Term};
use oxirs_core::model::NamedNode;
use std::sync::Arc;

#[test]
fn test_arithmetic_expressions() {
    let registry = ExtensionRegistry::new();
    let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

    // Bind some variables
    evaluator
        .binding_context_mut()
        .bind("x", Term::typed_literal("10", xsd::INTEGER).unwrap());
    evaluator
        .binding_context_mut()
        .bind("y", Term::typed_literal("5", xsd::INTEGER).unwrap());

    // Test addition
    let add_expr = Expression::Binary {
        op: BinaryOperator::Add,
        left: Box::new(Expression::Variable(Variable::new("x").unwrap())),
        right: Box::new(Expression::Variable(Variable::new("y").unwrap())),
    };

    let result = evaluator.evaluate(&add_expr).unwrap();
    assert_eq!(result, Term::typed_literal("15", xsd::INTEGER).unwrap());

    // Test subtraction
    let sub_expr = Expression::Binary {
        op: BinaryOperator::Subtract,
        left: Box::new(Expression::Variable(Variable::new("x").unwrap())),
        right: Box::new(Expression::Variable(Variable::new("y").unwrap())),
    };

    let result = evaluator.evaluate(&sub_expr).unwrap();
    assert_eq!(result, Term::typed_literal("5", xsd::INTEGER).unwrap());

    // Test multiplication
    let mul_expr = Expression::Binary {
        op: BinaryOperator::Multiply,
        left: Box::new(Expression::Variable(Variable::new("x").unwrap())),
        right: Box::new(Expression::Literal(Literal {
            value: "2".to_string(),
            language: None,
            datatype: Some(NamedNode::new(xsd::INTEGER).unwrap()),
        })),
    };

    let result = evaluator.evaluate(&mul_expr).unwrap();
    assert_eq!(result, Term::typed_literal("20", xsd::INTEGER).unwrap());
}

#[test]
fn test_comparison_expressions() {
    let registry = ExtensionRegistry::new();
    let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

    evaluator
        .binding_context_mut()
        .bind("a", Term::typed_literal("10", xsd::INTEGER).unwrap());
    evaluator
        .binding_context_mut()
        .bind("b", Term::typed_literal("20", xsd::INTEGER).unwrap());

    // Test less than
    let lt_expr = Expression::Binary {
        op: BinaryOperator::Less,
        left: Box::new(Expression::Variable(Variable::new("a").unwrap())),
        right: Box::new(Expression::Variable(Variable::new("b").unwrap())),
    };

    let result = evaluator.evaluate(&lt_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());

    // Test greater than
    let gt_expr = Expression::Binary {
        op: BinaryOperator::Greater,
        left: Box::new(Expression::Variable(Variable::new("a").unwrap())),
        right: Box::new(Expression::Variable(Variable::new("b").unwrap())),
    };

    let result = evaluator.evaluate(&gt_expr).unwrap();
    assert_eq!(result, Term::typed_literal("false", xsd::BOOLEAN).unwrap());

    // Test equality
    let eq_expr = Expression::Binary {
        op: BinaryOperator::Equal,
        left: Box::new(Expression::Variable(Variable::new("a").unwrap())),
        right: Box::new(Expression::Literal(Literal {
            value: "10".to_string(),
            language: None,
            datatype: Some(NamedNode::new(xsd::INTEGER).unwrap()),
        })),
    };

    let result = evaluator.evaluate(&eq_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());
}

#[test]
fn test_logical_expressions() {
    let registry = ExtensionRegistry::new();
    let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

    evaluator
        .binding_context_mut()
        .bind("p", Term::typed_literal("true", xsd::BOOLEAN).unwrap());
    evaluator
        .binding_context_mut()
        .bind("q", Term::typed_literal("false", xsd::BOOLEAN).unwrap());

    // Test AND
    let and_expr = Expression::Binary {
        op: BinaryOperator::And,
        left: Box::new(Expression::Variable(Variable::new("p").unwrap())),
        right: Box::new(Expression::Variable(Variable::new("q").unwrap())),
    };

    let result = evaluator.evaluate(&and_expr).unwrap();
    assert_eq!(result, Term::typed_literal("false", xsd::BOOLEAN).unwrap());

    // Test OR
    let or_expr = Expression::Binary {
        op: BinaryOperator::Or,
        left: Box::new(Expression::Variable(Variable::new("p").unwrap())),
        right: Box::new(Expression::Variable(Variable::new("q").unwrap())),
    };

    let result = evaluator.evaluate(&or_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());

    // Test NOT
    let not_expr = Expression::Unary {
        op: UnaryOperator::Not,
        expr: Box::new(Expression::Variable(Variable::new("q").unwrap())),
    };

    let result = evaluator.evaluate(&not_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());
}

#[test]
fn test_string_functions() {
    let registry = ExtensionRegistry::new();
    let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

    evaluator
        .binding_context_mut()
        .bind("name", Term::literal("hello world"));
    evaluator
        .binding_context_mut()
        .bind("url", Term::iri("http://example.org"));

    // Test STR function
    let str_expr = Expression::Function {
        name: "str".to_string(),
        args: vec![Expression::Variable(Variable::new("url").unwrap())],
    };

    let result = evaluator.evaluate(&str_expr).unwrap();
    assert_eq!(result, Term::literal("http://example.org"));

    // Test STRLEN function
    let strlen_expr = Expression::Function {
        name: "strlen".to_string(),
        args: vec![Expression::Variable(Variable::new("name").unwrap())],
    };

    let result = evaluator.evaluate(&strlen_expr).unwrap();
    assert_eq!(result, Term::typed_literal("11", xsd::INTEGER).unwrap());

    // Test UCASE function
    let ucase_expr = Expression::Function {
        name: "ucase".to_string(),
        args: vec![Expression::Variable(Variable::new("name").unwrap())],
    };

    let result = evaluator.evaluate(&ucase_expr).unwrap();
    assert_eq!(result, Term::literal("HELLO WORLD"));

    // Test SUBSTR function
    let substr_expr = Expression::Function {
        name: "substr".to_string(),
        args: vec![
            Expression::Variable(Variable::new("name").unwrap()),
            Expression::Literal(Literal {
                value: "1".to_string(),
                language: None,
                datatype: Some(NamedNode::new(xsd::INTEGER).unwrap()),
            }),
            Expression::Literal(Literal {
                value: "5".to_string(),
                language: None,
                datatype: Some(NamedNode::new(xsd::INTEGER).unwrap()),
            }),
        ],
    };

    let result = evaluator.evaluate(&substr_expr).unwrap();
    assert_eq!(result, Term::literal("hello"));
}

#[test]
fn test_type_checking_functions() {
    let registry = ExtensionRegistry::new();
    let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

    evaluator
        .binding_context_mut()
        .bind("iri", Term::iri("http://example.org"));
    evaluator
        .binding_context_mut()
        .bind("lit", Term::literal("test"));
    evaluator
        .binding_context_mut()
        .bind("blank", Term::blank_node("b1"));
    evaluator
        .binding_context_mut()
        .bind("num", Term::typed_literal("42", xsd::INTEGER).unwrap());

    // Test isIRI
    let is_iri_expr = Expression::Unary {
        op: UnaryOperator::IsIri,
        expr: Box::new(Expression::Variable(Variable::new("iri").unwrap())),
    };

    let result = evaluator.evaluate(&is_iri_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());

    // Test isLiteral
    let is_literal_expr = Expression::Unary {
        op: UnaryOperator::IsLiteral,
        expr: Box::new(Expression::Variable(Variable::new("lit").unwrap())),
    };

    let result = evaluator.evaluate(&is_literal_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());

    // Test isBlank
    let is_blank_expr = Expression::Unary {
        op: UnaryOperator::IsBlank,
        expr: Box::new(Expression::Variable(Variable::new("blank").unwrap())),
    };

    let result = evaluator.evaluate(&is_blank_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());

    // Test isNumeric
    let is_numeric_expr = Expression::Unary {
        op: UnaryOperator::IsNumeric,
        expr: Box::new(Expression::Variable(Variable::new("num").unwrap())),
    };

    let result = evaluator.evaluate(&is_numeric_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());
}

#[test]
fn test_conditional_expression() {
    let registry = ExtensionRegistry::new();
    let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

    evaluator
        .binding_context_mut()
        .bind("x", Term::typed_literal("10", xsd::INTEGER).unwrap());

    // Test IF expression
    let if_expr = Expression::Conditional {
        condition: Box::new(Expression::Binary {
            op: BinaryOperator::Greater,
            left: Box::new(Expression::Variable(Variable::new("x").unwrap())),
            right: Box::new(Expression::Literal(Literal {
                value: "5".to_string(),
                language: None,
                datatype: Some(NamedNode::new(xsd::INTEGER).unwrap()),
            })),
        }),
        then_expr: Box::new(Expression::Literal(Literal {
            value: "big".to_string(),
            language: None,
            datatype: None,
        })),
        else_expr: Box::new(Expression::Literal(Literal {
            value: "small".to_string(),
            language: None,
            datatype: None,
        })),
    };

    let result = evaluator.evaluate(&if_expr).unwrap();
    assert_eq!(result, Term::literal("big"));
}

#[test]
fn test_bound_expression() {
    let registry = ExtensionRegistry::new();
    let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

    evaluator
        .binding_context_mut()
        .bind("x", Term::literal("value"));

    // Test BOUND on bound variable
    let bound_expr = Expression::Bound(Variable::new("x").unwrap());
    let result = evaluator.evaluate(&bound_expr).unwrap();
    assert_eq!(result, Term::typed_literal("true", xsd::BOOLEAN).unwrap());

    // Test BOUND on unbound variable
    let unbound_expr = Expression::Bound(Variable::new("y").unwrap());
    let result = evaluator.evaluate(&unbound_expr).unwrap();
    assert_eq!(result, Term::typed_literal("false", xsd::BOOLEAN).unwrap());
}

#[test]
fn test_constructor_functions() {
    let registry = ExtensionRegistry::new();
    let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

    evaluator
        .binding_context_mut()
        .bind("str", Term::literal("http://example.org/resource"));
    evaluator
        .binding_context_mut()
        .bind("lang", Term::literal("en"));

    // Test IRI constructor
    let iri_expr = Expression::Function {
        name: "iri".to_string(),
        args: vec![Expression::Variable(Variable::new("str").unwrap())],
    };

    let result = evaluator.evaluate(&iri_expr).unwrap();
    assert!(result.is_iri());

    // Test STRLANG constructor
    let strlang_expr = Expression::Function {
        name: "strlang".to_string(),
        args: vec![
            Expression::Literal(Literal {
                value: "hello".to_string(),
                language: None,
                datatype: None,
            }),
            Expression::Variable(Variable::new("lang").unwrap()),
        ],
    };

    let result = evaluator.evaluate(&strlang_expr).unwrap();
    match result {
        Term::Literal(lit) => {
            assert_eq!(lit.lexical_form, "hello");
            assert_eq!(lit.language_tag, Some("en".to_string()));
        }
        _ => panic!("Expected language-tagged literal"),
    }
}
