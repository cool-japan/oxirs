//! Tests for the comprehensive term system

use oxirs_arq::term::{Term, LiteralValue, NumericValue, BindingContext, xsd, matches_pattern};
use oxirs_arq::algebra::{Term as AlgebraTerm, Iri, Literal};

#[test]
fn test_term_creation_and_conversion() {
    // Test IRI term
    let iri_term = Term::iri("http://example.org/resource");
    assert!(iri_term.is_iri());
    assert!(!iri_term.is_literal());
    
    // Test literal terms
    let simple_lit = Term::literal("hello world");
    assert!(simple_lit.is_literal());
    
    let typed_lit = Term::typed_literal("42", xsd::INTEGER).unwrap();
    assert!(typed_lit.is_literal());
    
    let lang_lit = Term::lang_literal("bonjour", "fr");
    assert!(lang_lit.is_literal());
    
    // Test blank node
    let blank = Term::blank_node("b1");
    assert!(blank.is_blank_node());
    
    // Test variable
    let var = Term::variable("x");
    assert!(var.is_variable());
    assert!(!var.is_ground());
}

#[test]
fn test_numeric_operations() {
    let int_term = Term::typed_literal("42", xsd::INTEGER).unwrap();
    let float_term = Term::typed_literal("3.14", xsd::FLOAT).unwrap();
    let double_term = Term::typed_literal("2.718", xsd::DOUBLE).unwrap();
    
    // Test numeric conversion
    let int_num = int_term.to_numeric().unwrap();
    assert_eq!(int_num, NumericValue::Integer(42));
    
    let float_num = float_term.to_numeric().unwrap();
    match float_num {
        NumericValue::Float(f) => assert!((f - 3.14).abs() < 0.001),
        _ => panic!("Expected float"),
    }
    
    // Test numeric promotion
    let (promoted_int, promoted_float) = int_num.promote_with(&float_num);
    match (promoted_int, promoted_float) {
        (NumericValue::Float(a), NumericValue::Float(b)) => {
            assert!((a - 42.0).abs() < 0.001);
            assert!((b - 3.14).abs() < 0.001);
        }
        _ => panic!("Expected both to be floats after promotion"),
    }
}

#[test]
fn test_effective_boolean_value() {
    // Boolean literals
    let true_term = Term::typed_literal("true", xsd::BOOLEAN).unwrap();
    assert_eq!(true_term.effective_boolean_value().unwrap(), true);
    
    let false_term = Term::typed_literal("false", xsd::BOOLEAN).unwrap();
    assert_eq!(false_term.effective_boolean_value().unwrap(), false);
    
    // Numeric literals
    let zero_term = Term::typed_literal("0", xsd::INTEGER).unwrap();
    assert_eq!(zero_term.effective_boolean_value().unwrap(), false);
    
    let nonzero_term = Term::typed_literal("42", xsd::INTEGER).unwrap();
    assert_eq!(nonzero_term.effective_boolean_value().unwrap(), true);
    
    // String literals
    let empty_string = Term::literal("");
    assert_eq!(empty_string.effective_boolean_value().unwrap(), false);
    
    let nonempty_string = Term::literal("test");
    assert_eq!(nonempty_string.effective_boolean_value().unwrap(), true);
    
    // Non-literals are truthy
    let iri = Term::iri("http://example.org");
    assert_eq!(iri.effective_boolean_value().unwrap(), true);
}

#[test]
fn test_term_ordering() {
    // Test SPARQL ordering: Unbound < Blank < IRI < Literal
    let var = Term::variable("x");
    let blank = Term::blank_node("b1");
    let iri = Term::iri("http://example.org");
    let lit = Term::literal("test");
    
    assert!(var < blank);
    assert!(blank < iri);
    assert!(iri < lit);
    
    // Test literal ordering
    let lit1 = Term::literal("apple");
    let lit2 = Term::literal("banana");
    assert!(lit1 < lit2);
    
    // Test language-tagged literals
    let en_lit = Term::lang_literal("hello", "en");
    let fr_lit = Term::lang_literal("hello", "fr");
    assert!(en_lit < fr_lit); // Language tags are compared
}

#[test]
fn test_datatype_parsing() {
    // Test various datatypes
    let date = Term::typed_literal("2023-01-01", xsd::DATE).unwrap();
    assert!(date.is_literal());
    
    let datetime = Term::typed_literal("2023-01-01T12:00:00Z", xsd::DATE_TIME).unwrap();
    assert!(datetime.is_literal());
    
    let hex = Term::typed_literal("48656c6c6f", xsd::HEX_BINARY).unwrap();
    assert!(hex.is_literal());
    
    let base64 = Term::typed_literal("SGVsbG8=", xsd::BASE64_BINARY).unwrap();
    assert!(base64.is_literal());
}

#[test]
fn test_binding_context() {
    let mut ctx = BindingContext::new();
    
    // Test basic binding
    let term1 = Term::literal("value1");
    ctx.bind("x", term1.clone());
    
    assert!(ctx.is_bound("x"));
    assert_eq!(ctx.get("x"), Some(&term1));
    assert!(!ctx.is_bound("y"));
    
    // Test scoping
    ctx.push_scope();
    
    let term2 = Term::literal("value2");
    ctx.bind("y", term2.clone());
    
    assert!(ctx.is_bound("x")); // Still visible from parent scope
    assert!(ctx.is_bound("y"));
    
    ctx.pop_scope();
    
    assert!(ctx.is_bound("x"));
    assert!(!ctx.is_bound("y")); // No longer visible after pop
    
    // Test variable listing
    let vars = ctx.variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains(&"x"));
}

#[test]
fn test_pattern_matching() {
    let mut ctx = BindingContext::new();
    
    // Variable pattern matches anything and binds
    let pattern = Term::variable("x");
    let term = Term::literal("test");
    
    assert!(matches_pattern(&pattern, &term, &mut ctx));
    assert_eq!(ctx.get("x"), Some(&term));
    
    // Second match with same variable checks equality
    let term2 = Term::literal("other");
    assert!(!matches_pattern(&pattern, &term2, &mut ctx));
    
    // Literal pattern requires exact match
    let lit_pattern = Term::literal("exact");
    let matching_term = Term::literal("exact");
    let non_matching_term = Term::literal("different");
    
    assert!(matches_pattern(&lit_pattern, &matching_term, &mut ctx));
    assert!(!matches_pattern(&lit_pattern, &non_matching_term, &mut ctx));
}

#[test]
fn test_algebra_term_conversion() {
    // Test conversion from algebra terms
    let algebra_iri = AlgebraTerm::Iri(Iri("http://example.org".to_string()));
    let term_iri = Term::from_algebra_term(&algebra_iri);
    assert!(term_iri.is_iri());
    
    let algebra_lit = AlgebraTerm::Literal(Literal {
        value: "42".to_string(),
        language: None,
        datatype: Some(Iri(xsd::INTEGER.to_string())),
    });
    let term_lit = Term::from_algebra_term(&algebra_lit);
    assert!(term_lit.is_literal());
    assert_eq!(term_lit.to_numeric().unwrap(), NumericValue::Integer(42));
    
    // Test conversion to algebra terms
    let term = Term::typed_literal("3.14", xsd::DOUBLE).unwrap();
    let algebra_term = term.to_algebra_term();
    
    match algebra_term {
        AlgebraTerm::Literal(lit) => {
            assert_eq!(lit.value, "3.14");
            assert_eq!(lit.datatype.as_ref().unwrap().0, xsd::DOUBLE);
        }
        _ => panic!("Expected literal"),
    }
}

#[test]
fn test_numeric_literal_creation() {
    // Test creating numeric literals
    let int_val = NumericValue::Integer(100);
    let int_term = int_val.to_term();
    assert!(int_term.is_literal());
    
    let decimal_val = NumericValue::Decimal(99.99);
    let decimal_term = decimal_val.to_term();
    assert!(decimal_term.is_literal());
    
    // Verify round-trip conversion
    let recovered = decimal_term.to_numeric().unwrap();
    match recovered {
        NumericValue::Decimal(d) => assert!((d - 99.99).abs() < 0.001),
        _ => panic!("Expected decimal"),
    }
}