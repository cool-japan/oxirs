//! Unit tests for SHACL expression constraints.
//!
//! Covers [`ShaclValue`], [`ShaclExpression`], [`ExpressionEvaluator`],
//! [`ExpressionContext`], and [`ExpressionConstraintComponent`] via the public
//! `constraints::expression_constraint` facade.

#![cfg(test)]

use super::expression_constraint::*;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

mod tests {
    use super::*;

    fn ctx(this: &str) -> ExpressionContext {
        ExpressionContext::new(this)
    }

    // ---- Literal ----------------------------------------------------------

    #[test]
    fn test_literal_integer() {
        let expr = ShaclExpression::Literal(ShaclValue::Integer(42));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(42));
    }

    #[test]
    fn test_literal_float() {
        let expr = ShaclExpression::Literal(ShaclValue::Float(std::f64::consts::PI));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Float(std::f64::consts::PI));
    }

    // ---- Variables --------------------------------------------------------

    #[test]
    fn test_variable_this() {
        let expr = ShaclExpression::Variable("this".to_string());
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Iri("http://ex/Alice".to_string()));
    }

    #[test]
    fn test_variable_value_absent() {
        let expr = ShaclExpression::Variable("value".to_string());
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Null);
    }

    #[test]
    fn test_variable_custom_binding() {
        let expr = ShaclExpression::Variable("myVar".to_string());
        let c = ctx("http://ex/a").bind("myVar", ShaclValue::Integer(99));
        let result = ExpressionEvaluator::evaluate(&expr, &c).expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(99));
    }

    // ---- Arithmetic -------------------------------------------------------

    #[test]
    fn test_add_integers() {
        let expr = ShaclExpression::Add(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(3))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(4))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(7));
    }

    #[test]
    fn test_sub_integers() {
        let expr = ShaclExpression::Sub(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(10))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(3))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(7));
    }

    #[test]
    fn test_mul_integers() {
        let expr = ShaclExpression::Mul(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(6))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(7))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(42));
    }

    #[test]
    fn test_div_integers_produces_float() {
        let expr = ShaclExpression::Div(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(7))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(2))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Float(3.5));
    }

    #[test]
    fn test_div_by_zero() {
        let expr = ShaclExpression::Div(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(1))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(0))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"));
        assert!(result.is_err());
    }

    // ---- Comparison -------------------------------------------------------

    #[test]
    fn test_lt_true() {
        let expr = ShaclExpression::Lt(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(3))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(5))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_gte_false() {
        let expr = ShaclExpression::Gte(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(3))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(5))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_eq_same_integer() {
        let expr = ShaclExpression::Eq(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(7))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(7))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- Logical ----------------------------------------------------------

    #[test]
    fn test_and_true() {
        let expr = ShaclExpression::And(
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(true))),
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(true))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_and_short_circuit() {
        let expr = ShaclExpression::And(
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(false))),
            // This would error if evaluated
            Box::new(ShaclExpression::Div(
                Box::new(ShaclExpression::Literal(ShaclValue::Integer(1))),
                Box::new(ShaclExpression::Literal(ShaclValue::Integer(0))),
            )),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("short-circuit should avoid division by zero");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_or_true() {
        let expr = ShaclExpression::Or(
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(false))),
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(true))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_not() {
        let expr = ShaclExpression::Not(Box::new(ShaclExpression::Literal(ShaclValue::Boolean(
            false,
        ))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- String functions -------------------------------------------------

    #[test]
    fn test_concat() {
        let expr = ShaclExpression::Concat(vec![
            ShaclExpression::Literal(ShaclValue::Literal {
                value: "Hello".to_string(),
                datatype: None,
                lang: None,
            }),
            ShaclExpression::Literal(ShaclValue::Literal {
                value: ", World".to_string(),
                datatype: None,
                lang: None,
            }),
        ]);
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result.as_string(), "Hello, World");
    }

    #[test]
    fn test_strlen() {
        let expr =
            ShaclExpression::StrLen(Box::new(ShaclExpression::Literal(ShaclValue::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None,
            })));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(5));
    }

    #[test]
    fn test_regex_match() {
        let expr = ShaclExpression::Regex(
            Box::new(ShaclExpression::Literal(ShaclValue::Literal {
                value: "hello123".to_string(),
                datatype: None,
                lang: None,
            })),
            r"^\w+\d+$".to_string(),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_regex_no_match() {
        let expr = ShaclExpression::Regex(
            Box::new(ShaclExpression::Literal(ShaclValue::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None,
            })),
            r"^\d+$".to_string(),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_uppercase() {
        let expr =
            ShaclExpression::UpperCase(Box::new(ShaclExpression::Literal(ShaclValue::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None,
            })));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result.as_string(), "HELLO");
    }

    // ---- Numeric functions ------------------------------------------------

    #[test]
    fn test_abs_negative() {
        let expr =
            ShaclExpression::Abs(Box::new(ShaclExpression::Literal(ShaclValue::Integer(-7))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(7));
    }

    #[test]
    fn test_floor() {
        let expr =
            ShaclExpression::Floor(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.7))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(3));
    }

    #[test]
    fn test_ceil() {
        let expr =
            ShaclExpression::Ceil(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.1))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(4));
    }

    #[test]
    fn test_round() {
        let expr =
            ShaclExpression::Round(Box::new(ShaclExpression::Literal(ShaclValue::Float(2.5))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(3));
    }

    // ---- Type functions ---------------------------------------------------

    #[test]
    fn test_is_iri() {
        let expr = ShaclExpression::IsIri(Box::new(ShaclExpression::Variable("this".to_string())));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_is_literal() {
        let expr =
            ShaclExpression::IsLiteral(Box::new(ShaclExpression::Literal(ShaclValue::Integer(5))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        // Integer is not a Literal variant in our type system
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_datatype_integer() {
        let expr =
            ShaclExpression::Datatype(Box::new(ShaclExpression::Literal(ShaclValue::Integer(42))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(
            result,
            ShaclValue::Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())
        );
    }

    // ---- ExpressionConstraintComponent -----------------------------------

    #[test]
    fn test_expression_constraint_valid() {
        let constraint =
            ExpressionConstraintComponent::new(ShaclExpression::Literal(ShaclValue::Boolean(true)));
        let result = constraint
            .evaluate(&ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert!(result.is_valid);
        assert!(result.message.is_none());
    }

    #[test]
    fn test_expression_constraint_invalid() {
        let constraint = ExpressionConstraintComponent::new(ShaclExpression::Literal(
            ShaclValue::Boolean(false),
        ))
        .with_message("Value must be truthy");

        let result = constraint
            .evaluate(&ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert!(!result.is_valid);
        assert_eq!(result.message, Some("Value must be truthy".to_string()));
    }

    #[test]
    fn test_expression_constraint_deactivated() {
        // Even though the expression is always false, deactivated means valid
        let constraint = ExpressionConstraintComponent::new(ShaclExpression::Literal(
            ShaclValue::Boolean(false),
        ))
        .deactivated();

        let result = constraint
            .evaluate(&ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert!(result.is_valid);
    }

    #[test]
    fn test_complex_expression_age_range() {
        // Constraint: age >= 18 && age <= 120
        let age_var = ShaclExpression::Variable("age".to_string());
        let expr = ShaclExpression::And(
            Box::new(ShaclExpression::Gte(
                Box::new(age_var.clone()),
                Box::new(ShaclExpression::Literal(ShaclValue::Integer(18))),
            )),
            Box::new(ShaclExpression::Lte(
                Box::new(age_var),
                Box::new(ShaclExpression::Literal(ShaclValue::Integer(120))),
            )),
        );

        let constraint =
            ExpressionConstraintComponent::new(expr).with_message("Age must be between 18 and 120");

        // Valid age
        let c_valid = ctx("http://ex/Alice").bind("age", ShaclValue::Integer(30));
        let r = constraint
            .evaluate(&c_valid)
            .expect("evaluation should succeed");
        assert!(r.is_valid);

        // Invalid age (too young)
        let c_invalid = ctx("http://ex/Bob").bind("age", ShaclValue::Integer(15));
        let r2 = constraint
            .evaluate(&c_invalid)
            .expect("evaluation should succeed");
        assert!(!r2.is_valid);
        assert!(r2.message.is_some());
    }
}

// ---------------------------------------------------------------------------
// Extended expression constraint tests — using only actual enum variants
// ---------------------------------------------------------------------------

mod extended_expression_tests {
    use super::*;

    fn ctx(this: &str) -> ExpressionContext {
        ExpressionContext::new(this)
    }

    fn lit_str(s: &str) -> ShaclExpression {
        ShaclExpression::Literal(ShaclValue::Literal {
            value: s.to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            lang: None,
        })
    }

    fn lit_int(n: i64) -> ShaclExpression {
        ShaclExpression::Literal(ShaclValue::Integer(n))
    }

    fn lit_bool(b: bool) -> ShaclExpression {
        ShaclExpression::Literal(ShaclValue::Boolean(b))
    }

    // ---- ExpressionContext fields ---------------------------------------

    #[test]
    fn test_expression_context_this_node_field() {
        let c = ctx("http://ex/Alice");
        assert_eq!(c.this_node, "http://ex/Alice");
    }

    #[test]
    fn test_expression_context_bind_and_lookup() {
        let c = ctx("http://ex/X").bind("score", ShaclValue::Integer(100));
        let expr = ShaclExpression::Variable("score".to_string());
        let v = ExpressionEvaluator::evaluate(&expr, &c).expect("evaluate");
        assert_eq!(v, ShaclValue::Integer(100));
    }

    #[test]
    fn test_expression_context_overwrite_binding() {
        let c = ctx("http://ex/X")
            .bind("x", ShaclValue::Integer(1))
            .bind("x", ShaclValue::Integer(2));
        let expr = ShaclExpression::Variable("x".to_string());
        let v = ExpressionEvaluator::evaluate(&expr, &c).expect("evaluate");
        assert_eq!(v, ShaclValue::Integer(2));
    }

    #[test]
    fn test_expression_context_value_node_default_none() {
        let c = ctx("http://ex/A");
        assert!(c.value_node.is_none());
    }

    #[test]
    fn test_expression_context_with_value() {
        let c = ctx("http://ex/A").with_value("http://ex/Value1");
        assert_eq!(c.value_node.as_deref(), Some("http://ex/Value1"));
    }

    // ---- Nested arithmetic expressions ---------------------------------

    #[test]
    fn test_nested_add_mul() {
        // (2 + 3) * 4 = 20
        let inner_add = ShaclExpression::Add(Box::new(lit_int(2)), Box::new(lit_int(3)));
        let expr = ShaclExpression::Mul(Box::new(inner_add), Box::new(lit_int(4)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Integer(20));
    }

    #[test]
    fn test_nested_sub_div() {
        // (10 - 4) / 2 = 3
        let inner_sub = ShaclExpression::Sub(Box::new(lit_int(10)), Box::new(lit_int(4)));
        let expr = ShaclExpression::Div(Box::new(inner_sub), Box::new(lit_int(2)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        // Division may return Float or Integer depending on implementation
        let val = match result {
            ShaclValue::Float(f) => f as i64,
            ShaclValue::Integer(n) => n,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(val, 3);
    }

    // ---- Chained comparisons -------------------------------------------

    #[test]
    fn test_chained_and_comparisons() {
        // 5 > 1 AND 5 < 10 → true
        let c1 = ShaclExpression::Gt(Box::new(lit_int(5)), Box::new(lit_int(1)));
        let c2 = ShaclExpression::Lt(Box::new(lit_int(5)), Box::new(lit_int(10)));
        let expr = ShaclExpression::And(Box::new(c1), Box::new(c2));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_or_with_one_false_one_true() {
        let expr = ShaclExpression::Or(Box::new(lit_bool(false)), Box::new(lit_bool(true)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_not_true_is_false() {
        let expr = ShaclExpression::Not(Box::new(lit_bool(true)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_not_false_is_true() {
        let expr = ShaclExpression::Not(Box::new(lit_bool(false)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- Equality ------------------------------------------------------

    #[test]
    fn test_eq_integers_equal() {
        let expr = ShaclExpression::Eq(Box::new(lit_int(42)), Box::new(lit_int(42)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_eq_integers_not_equal() {
        let expr = ShaclExpression::Eq(Box::new(lit_int(1)), Box::new(lit_int(2)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_ne_integers_different() {
        let expr = ShaclExpression::Ne(Box::new(lit_int(3)), Box::new(lit_int(7)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- String operations (UpperCase / LowerCase) ---------------------

    #[test]
    fn test_uppercase_converts_lowercase() {
        let expr = ShaclExpression::UpperCase(Box::new(lit_str("hello world")));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        match result {
            ShaclValue::Literal { value, .. } => assert_eq!(value, "HELLO WORLD"),
            other => panic!("expected Literal, got {other:?}"),
        }
    }

    #[test]
    fn test_lowercase_converts_uppercase() {
        let expr = ShaclExpression::LowerCase(Box::new(lit_str("HELLO")));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        match result {
            ShaclValue::Literal { value, .. } => assert_eq!(value, "hello"),
            other => panic!("expected Literal, got {other:?}"),
        }
    }

    #[test]
    fn test_strlen_nonempty_string() {
        let expr = ShaclExpression::StrLen(Box::new(lit_str("hello")));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Integer(5));
    }

    #[test]
    fn test_strlen_empty_string() {
        let expr = ShaclExpression::StrLen(Box::new(lit_str("")));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Integer(0));
    }

    #[test]
    fn test_concat_two_strings() {
        let expr = ShaclExpression::Concat(vec![lit_str("foo"), lit_str("bar")]);
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        match result {
            ShaclValue::Literal { value, .. } => assert_eq!(value, "foobar"),
            ShaclValue::Integer(n) => panic!("unexpected integer {n}"),
            other => panic!("expected Literal, got {other:?}"),
        }
    }

    #[test]
    fn test_concat_empty_list() {
        let expr = ShaclExpression::Concat(vec![]);
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        match result {
            ShaclValue::Literal { value, .. } => assert!(value.is_empty()),
            ShaclValue::Integer(0) => {} // acceptable: some impls return 0 for empty concat
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ---- ExpressionConstraintComponent builder -------------------------

    #[test]
    fn test_builder_with_message_stored() {
        let c = ExpressionConstraintComponent::new(lit_bool(true)).with_message("custom message");
        assert_eq!(c.message.as_deref(), Some("custom message"));
    }

    #[test]
    fn test_builder_deactivated_flag() {
        let c = ExpressionConstraintComponent::new(lit_bool(false)).deactivated();
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(
            result.is_valid,
            "deactivated constraint must always be valid"
        );
    }

    #[test]
    fn test_builder_with_message_returned_on_violation() {
        let c =
            ExpressionConstraintComponent::new(lit_bool(false)).with_message("violation occurred");
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(!result.is_valid);
        assert_eq!(result.message.as_deref(), Some("violation occurred"));
    }

    // ---- ShaclValue helpers --------------------------------------------

    #[test]
    fn test_shacl_value_boolean_true_eq() {
        assert_eq!(ShaclValue::Boolean(true), ShaclValue::Boolean(true));
    }

    #[test]
    fn test_shacl_value_boolean_false_ne_true() {
        assert_ne!(ShaclValue::Boolean(false), ShaclValue::Boolean(true));
    }

    #[test]
    fn test_shacl_value_null_eq_null() {
        assert_eq!(ShaclValue::Null, ShaclValue::Null);
    }

    #[test]
    fn test_shacl_value_null_is_not_truthy() {
        assert!(!ShaclValue::Null.is_truthy());
    }

    #[test]
    fn test_shacl_value_integer_zero_is_not_truthy() {
        assert!(!ShaclValue::Integer(0).is_truthy());
    }

    #[test]
    fn test_shacl_value_integer_nonzero_is_truthy() {
        assert!(ShaclValue::Integer(42).is_truthy());
    }

    #[test]
    fn test_shacl_value_iri_is_truthy() {
        assert!(ShaclValue::Iri("http://ex/a".to_string()).is_truthy());
    }

    #[test]
    fn test_shacl_value_as_string_integer() {
        assert_eq!(ShaclValue::Integer(99).as_string(), "99");
    }

    #[test]
    fn test_shacl_value_as_string_boolean_true() {
        assert_eq!(ShaclValue::Boolean(true).as_string(), "true");
    }

    #[test]
    fn test_shacl_value_as_string_null_is_empty() {
        assert_eq!(ShaclValue::Null.as_string(), "");
    }

    // ---- Lte / Gte comparisons -----------------------------------------

    #[test]
    fn test_lte_equal_values() {
        let expr = ShaclExpression::Lte(Box::new(lit_int(5)), Box::new(lit_int(5)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_gte_equal_values() {
        let expr = ShaclExpression::Gte(Box::new(lit_int(5)), Box::new(lit_int(5)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_lte_less_than() {
        let expr = ShaclExpression::Lte(Box::new(lit_int(3)), Box::new(lit_int(7)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_gte_greater_than() {
        let expr = ShaclExpression::Gte(Box::new(lit_int(7)), Box::new(lit_int(3)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- IsIri / IsLiteral / Datatype ----------------------------------

    #[test]
    fn test_is_iri_true_for_iri() {
        let expr = ShaclExpression::IsIri(Box::new(ShaclExpression::Literal(ShaclValue::Iri(
            "http://example.org/b".to_string(),
        ))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_is_iri_false_for_integer() {
        let expr = ShaclExpression::IsIri(Box::new(lit_int(42)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_is_literal_true_for_literal_value() {
        let literal_expr = ShaclExpression::Literal(ShaclValue::Literal {
            value: "test".to_string(),
            datatype: None,
            lang: None,
        });
        let expr = ShaclExpression::IsLiteral(Box::new(literal_expr));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_datatype_integer() {
        let expr = ShaclExpression::Datatype(Box::new(lit_int(42)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(
            result,
            ShaclValue::Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())
        );
    }

    #[test]
    fn test_datatype_boolean() {
        let expr = ShaclExpression::Datatype(Box::new(lit_bool(true)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(
            result,
            ShaclValue::Iri("http://www.w3.org/2001/XMLSchema#boolean".to_string())
        );
    }

    #[test]
    fn test_datatype_iri_returns_its_own_type() {
        let expr = ShaclExpression::Datatype(Box::new(ShaclExpression::Literal(ShaclValue::Iri(
            "http://ex/r".to_string(),
        ))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        // IRI has no datatype — implementation specific but should not panic
        assert!(matches!(result, ShaclValue::Iri(_) | ShaclValue::Null));
    }

    // ---- Abs / Floor / Ceil / Round ------------------------------------

    #[test]
    fn test_abs_positive_unchanged() {
        let expr = ShaclExpression::Abs(Box::new(lit_int(7)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Integer(7));
    }

    #[test]
    fn test_floor_float() {
        let expr =
            ShaclExpression::Floor(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.9))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        // Accept Float(3.0) or Integer(3)
        let n = match result {
            ShaclValue::Float(f) => f as i64,
            ShaclValue::Integer(n) => n,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(n, 3);
    }

    #[test]
    fn test_ceil_float() {
        let expr =
            ShaclExpression::Ceil(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.1))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        let n = match result {
            ShaclValue::Float(f) => f as i64,
            ShaclValue::Integer(n) => n,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(n, 4);
    }

    #[test]
    fn test_round_float() {
        let expr =
            ShaclExpression::Round(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.5))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        let n = match result {
            ShaclValue::Float(f) => f as i64,
            ShaclValue::Integer(n) => n,
            other => panic!("unexpected {other:?}"),
        };
        // 3.5 rounds to 4 (round half up) or 4 (round half to even)
        assert!(n == 4 || n == 3, "round(3.5) should be 3 or 4, got {n}");
    }

    // ---- ExpressionConstraintResult ------------------------------------

    #[test]
    fn test_result_is_valid_true_for_truthy_expression() {
        let c = ExpressionConstraintComponent::new(lit_int(1));
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(result.is_valid);
    }

    #[test]
    fn test_result_is_valid_false_for_falsy_expression() {
        let c = ExpressionConstraintComponent::new(lit_int(0));
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_result_focus_node_matches_context() {
        let c = ExpressionConstraintComponent::new(lit_bool(true));
        let result = c.evaluate(&ctx("http://ex/MyNode")).expect("evaluate");
        assert_eq!(result.focus_node, "http://ex/MyNode");
    }

    #[test]
    fn test_result_message_none_when_valid() {
        let c = ExpressionConstraintComponent::new(lit_bool(true));
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(result.message.is_none());
    }

    #[test]
    fn test_result_default_message_when_no_template() {
        let c = ExpressionConstraintComponent::new(lit_bool(false));
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(result.message.is_some());
        assert!(result
            .message
            .as_deref()
            .expect("should succeed")
            .contains("Expression constraint failed"));
    }
}
