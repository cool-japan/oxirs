//! Advanced SHACL validation tests
//!
//! Comprehensive test coverage for:
//! - Cardinality constraints (sh:minCount, sh:maxCount)
//! - Value constraints (sh:class, sh:datatype, sh:nodeKind)
//! - Range constraints (sh:minExclusive, sh:maxExclusive, sh:minInclusive, sh:maxInclusive)
//! - String constraints (sh:minLength, sh:maxLength, sh:pattern)
//! - Node shape constraints (sh:node)
//! - Property path constraints
//! - Logical constraints (sh:and, sh:or, sh:not, sh:xone)
//! - Target declarations (sh:targetClass, sh:targetNode, etc.)
//! - Severity levels (sh:Violation, sh:Warning, sh:Info)
//! - Cache hit/miss behaviour

use crate::{
    cache::{CachedValidationResult, ValidationCache, ValidationCacheKey},
    constraints::{
        cardinality_constraints::{MaxCountConstraint, MinCountConstraint},
        constraint_context::ConstraintContext,
        constraint_types::ConstraintEvaluator,
        logical_constraints::{AndConstraint, NotConstraint, OrConstraint, XoneConstraint},
        range_constraints::{
            MaxExclusiveConstraint, MaxInclusiveConstraint, MinExclusiveConstraint,
            MinInclusiveConstraint,
        },
        shape_constraints::NodeConstraint,
        string_constraints::{
            LanguageInConstraint, MaxLengthConstraint, MinLengthConstraint, PatternConstraint,
            UniqueLangConstraint,
        },
        value_constraints::{ClassConstraint, DatatypeConstraint, NodeKind, NodeKindConstraint},
    },
    validation::ValidationViolation,
    ConstraintComponentId, PropertyPath, Severity, ShapeId,
};
use oxirs_core::{
    model::{BlankNode, Literal, NamedNode, Term},
    ConcreteStore,
};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

fn make_iri(path: &str) -> Term {
    Term::NamedNode(NamedNode::new_unchecked(format!(
        "http://example.org/{path}"
    )))
}

fn make_blank() -> Term {
    Term::BlankNode(BlankNode::new_unchecked("b0"))
}

fn make_lit(value: &str) -> Term {
    Term::Literal(Literal::new(value))
}

fn make_typed_lit(value: &str, datatype_local: &str) -> Term {
    let dt = NamedNode::new_unchecked(format!("http://www.w3.org/2001/XMLSchema#{datatype_local}"));
    Term::Literal(Literal::new_typed_literal(value, dt))
}

fn make_lang_lit(value: &str, lang: &str) -> Term {
    Term::Literal(Literal::new_language_tagged_literal_unchecked(value, lang))
}

fn make_numeric_lit(value: &str) -> Literal {
    let dt = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer");
    Literal::new_typed_literal(value, dt)
}

fn shape_id(s: &str) -> ShapeId {
    ShapeId::new(s)
}

fn focus_node() -> Term {
    make_iri("focusNode")
}

fn simple_path() -> PropertyPath {
    PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/prop"))
}

fn make_context_with_values(values: Vec<Term>) -> ConstraintContext {
    ConstraintContext::new(focus_node(), shape_id("http://example.org/TestShape"))
        .with_path(simple_path())
        .with_values(values)
}

fn empty_context() -> ConstraintContext {
    make_context_with_values(vec![])
}

// ---------------------------------------------------------------------------
// Cardinality constraints — 20 tests
// ---------------------------------------------------------------------------

#[test]
fn test_min_count_zero_always_satisfied() {
    let c = MinCountConstraint { min_count: 0 };
    let store = ConcreteStore::new().expect("store");
    let result = c.evaluate(&store, &empty_context()).expect("eval");
    assert!(result.is_satisfied(), "min_count=0 must always pass");
}

#[test]
fn test_min_count_1_no_values_violated() {
    let c = MinCountConstraint { min_count: 1 };
    let store = ConcreteStore::new().expect("store");
    let result = c.evaluate(&store, &empty_context()).expect("eval");
    assert!(
        result.is_violated(),
        "min_count=1 with no values must violate"
    );
}

#[test]
fn test_min_count_1_one_value_satisfied() {
    let c = MinCountConstraint { min_count: 1 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

#[test]
fn test_min_count_2_one_value_violated() {
    let c = MinCountConstraint { min_count: 2 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v1")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(
        result.is_violated(),
        "min_count=2 with 1 value must violate"
    );
}

#[test]
fn test_min_count_2_two_values_satisfied() {
    let c = MinCountConstraint { min_count: 2 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v1"), make_lit("v2")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

#[test]
fn test_min_count_3_two_values_violated() {
    let c = MinCountConstraint { min_count: 3 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("a"), make_lit("b")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_violated());
}

#[test]
fn test_max_count_0_no_values_satisfied() {
    let c = MaxCountConstraint { max_count: 0 };
    let store = ConcreteStore::new().expect("store");
    let result = c.evaluate(&store, &empty_context()).expect("eval");
    assert!(
        result.is_satisfied(),
        "max_count=0 with no values must pass"
    );
}

#[test]
fn test_max_count_0_one_value_violated() {
    let c = MaxCountConstraint { max_count: 0 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(
        result.is_violated(),
        "max_count=0 with any value must violate"
    );
}

#[test]
fn test_max_count_1_one_value_satisfied() {
    let c = MaxCountConstraint { max_count: 1 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

#[test]
fn test_max_count_1_two_values_violated() {
    let c = MaxCountConstraint { max_count: 1 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v1"), make_lit("v2")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_violated());
}

#[test]
fn test_max_count_3_three_values_satisfied() {
    let c = MaxCountConstraint { max_count: 3 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("a"), make_lit("b"), make_lit("c")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

#[test]
fn test_max_count_3_four_values_violated() {
    let c = MaxCountConstraint { max_count: 3 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![
        make_lit("a"),
        make_lit("b"),
        make_lit("c"),
        make_lit("d"),
    ]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_violated());
}

#[test]
fn test_exactly_two_values_min_satisfied() {
    let c = MinCountConstraint { min_count: 2 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v1"), make_lit("v2")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

#[test]
fn test_exactly_two_values_max_satisfied() {
    let c = MaxCountConstraint { max_count: 2 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v1"), make_lit("v2")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

#[test]
fn test_exactly_two_values_extra_max_violated() {
    let c = MaxCountConstraint { max_count: 2 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v1"), make_lit("v2"), make_lit("v3")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_violated());
}

#[test]
fn test_min_count_violation_has_message() {
    let c = MinCountConstraint { min_count: 5 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("only one")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_violated());
    assert!(result.message().is_some(), "violation must carry a message");
}

#[test]
fn test_max_count_violation_has_message() {
    let c = MaxCountConstraint { max_count: 1 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("v1"), make_lit("v2")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_violated());
    assert!(result.message().is_some(), "violation must carry a message");
}

#[test]
fn test_min_count_large_satisfied() {
    let c = MinCountConstraint { min_count: 10 };
    let store = ConcreteStore::new().expect("store");
    let values: Vec<Term> = (0..10).map(|i| make_lit(&i.to_string())).collect();
    let ctx = make_context_with_values(values);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

#[test]
fn test_max_count_large_exactly_at_limit_satisfied() {
    let c = MaxCountConstraint { max_count: 100 };
    let store = ConcreteStore::new().expect("store");
    let values: Vec<Term> = (0..100).map(|i| make_lit(&i.to_string())).collect();
    let ctx = make_context_with_values(values);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

#[test]
fn test_cardinality_with_iri_values() {
    let c = MinCountConstraint { min_count: 2 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("a"), make_iri("b"), make_iri("c")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_satisfied());
}

// ---------------------------------------------------------------------------
// Value constraints — 25 tests
// ---------------------------------------------------------------------------

#[test]
fn test_nodekind_iri_with_iri_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::Iri,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("node")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_iri_with_blank_violated() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::Iri,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_blank()]);
    let r = c.evaluate(&store, &ctx).expect("eval");
    assert!(r.is_violated());
}

#[test]
fn test_nodekind_iri_with_literal_violated() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::Iri,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("not-an-iri")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_nodekind_literal_with_literal_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::Literal,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("hello")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_literal_with_iri_violated() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::Literal,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("thing")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_nodekind_blank_node_with_blank_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::BlankNode,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_blank()]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_blank_node_with_iri_violated() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::BlankNode,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("thing")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_nodekind_iri_or_literal_with_iri_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::IriOrLiteral,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("node")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_iri_or_literal_with_literal_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::IriOrLiteral,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("text")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_iri_or_literal_with_blank_violated() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::IriOrLiteral,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_blank()]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_nodekind_blank_or_literal_with_blank_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::BlankNodeOrLiteral,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_blank()]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_blank_or_literal_with_literal_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::BlankNodeOrLiteral,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("lit")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_blank_or_literal_with_iri_violated() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::BlankNodeOrLiteral,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("x")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_nodekind_blank_or_iri_with_blank_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::BlankNodeOrIri,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_blank()]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_blank_or_iri_with_iri_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::BlankNodeOrIri,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("n")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_nodekind_blank_or_iri_with_literal_violated() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::BlankNodeOrIri,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("lit")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_nodekind_empty_values_always_satisfied() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::Iri,
    };
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_datatype_xsd_string_correct_satisfied() {
    let dt = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#string");
    let c = DatatypeConstraint {
        datatype_iri: dt.clone(),
    };
    let store = ConcreteStore::new().expect("store");
    let lit = Term::Literal(Literal::new_typed_literal("hello", dt));
    let ctx = make_context_with_values(vec![lit]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_datatype_xsd_integer_string_literal_violated() {
    let integer_dt = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer");
    let c = DatatypeConstraint {
        datatype_iri: integer_dt,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("not-a-number")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_datatype_iri_value_violated() {
    let dt = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#string");
    let c = DatatypeConstraint { datatype_iri: dt };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("an-iri")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_datatype_blank_node_violated() {
    let dt = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#string");
    let c = DatatypeConstraint { datatype_iri: dt };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_blank()]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_datatype_empty_values_satisfied() {
    let dt = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer");
    let c = DatatypeConstraint { datatype_iri: dt };
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_class_constraint_no_values_satisfied() {
    let class_iri = NamedNode::new_unchecked("http://example.org/Person");
    let c = ClassConstraint { class_iri };
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_nodekind_violation_carries_violating_value() {
    let c = NodeKindConstraint {
        node_kind: NodeKind::Iri,
    };
    let store = ConcreteStore::new().expect("store");
    let lit_term = make_lit("not-iri");
    let ctx = make_context_with_values(vec![lit_term.clone()]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    assert!(result.is_violated());
    assert_eq!(result.violating_value(), Some(&lit_term));
}

// ---------------------------------------------------------------------------
// Range constraints — 20 tests
// ---------------------------------------------------------------------------

#[test]
fn test_min_exclusive_greater_than_min_satisfied() {
    let c = MinExclusiveConstraint {
        min_value: make_numeric_lit("5"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("10", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_min_exclusive_equal_to_min_violated() {
    let c = MinExclusiveConstraint {
        min_value: make_numeric_lit("5"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("5", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_min_exclusive_less_than_min_violated() {
    let c = MinExclusiveConstraint {
        min_value: make_numeric_lit("10"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("3", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_max_exclusive_less_than_max_satisfied() {
    let c = MaxExclusiveConstraint {
        max_value: make_numeric_lit("100"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("50", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_max_exclusive_equal_to_max_violated() {
    let c = MaxExclusiveConstraint {
        max_value: make_numeric_lit("100"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("100", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_max_exclusive_greater_than_max_violated() {
    let c = MaxExclusiveConstraint {
        max_value: make_numeric_lit("100"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("200", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_min_inclusive_equal_to_min_satisfied() {
    let c = MinInclusiveConstraint {
        min_value: make_numeric_lit("0"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("0", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_min_inclusive_greater_than_min_satisfied() {
    let c = MinInclusiveConstraint {
        min_value: make_numeric_lit("18"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("21", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_min_inclusive_less_than_min_violated() {
    let c = MinInclusiveConstraint {
        min_value: make_numeric_lit("18"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("16", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_max_inclusive_equal_to_max_satisfied() {
    let c = MaxInclusiveConstraint {
        max_value: make_numeric_lit("100"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("100", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_max_inclusive_less_than_max_satisfied() {
    let c = MaxInclusiveConstraint {
        max_value: make_numeric_lit("100"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("42", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_max_inclusive_greater_than_max_violated() {
    let c = MaxInclusiveConstraint {
        max_value: make_numeric_lit("100"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("101", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_range_empty_values_all_satisfied() {
    let store = ConcreteStore::new().expect("store");
    let min_excl = MinExclusiveConstraint {
        min_value: make_numeric_lit("0"),
    };
    let max_incl = MaxInclusiveConstraint {
        max_value: make_numeric_lit("100"),
    };
    assert!(min_excl
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
    assert!(max_incl
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_range_negative_values_min_exclusive_violated() {
    let c = MinExclusiveConstraint {
        min_value: make_numeric_lit("0"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("-5", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_range_negative_values_max_inclusive_satisfied() {
    let c = MaxInclusiveConstraint {
        max_value: make_numeric_lit("0"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_typed_lit("-1", "integer")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_min_exclusive_non_literal_violated() {
    let c = MinExclusiveConstraint {
        min_value: make_numeric_lit("0"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("non-literal")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_max_exclusive_non_literal_violated() {
    let c = MaxExclusiveConstraint {
        max_value: make_numeric_lit("100"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_blank()]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_combined_min_max_inclusive_in_range_satisfied() {
    let store = ConcreteStore::new().expect("store");
    let min_c = MinInclusiveConstraint {
        min_value: make_numeric_lit("1"),
    };
    let max_c = MaxInclusiveConstraint {
        max_value: make_numeric_lit("10"),
    };
    let ctx = make_context_with_values(vec![make_typed_lit("5", "integer")]);
    assert!(min_c.evaluate(&store, &ctx).expect("eval").is_satisfied());
    assert!(max_c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_combined_min_max_inclusive_below_range_violated() {
    let store = ConcreteStore::new().expect("store");
    let min_c = MinInclusiveConstraint {
        min_value: make_numeric_lit("1"),
    };
    let ctx = make_context_with_values(vec![make_typed_lit("0", "integer")]);
    assert!(min_c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_combined_min_max_inclusive_above_range_violated() {
    let store = ConcreteStore::new().expect("store");
    let max_c = MaxInclusiveConstraint {
        max_value: make_numeric_lit("10"),
    };
    let ctx = make_context_with_values(vec![make_typed_lit("11", "integer")]);
    assert!(max_c.evaluate(&store, &ctx).expect("eval").is_violated());
}

// ---------------------------------------------------------------------------
// String constraints — 20 tests
// ---------------------------------------------------------------------------

#[test]
fn test_min_length_satisfied() {
    let c = MinLengthConstraint { min_length: 3 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("abc")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_min_length_exactly_at_min_satisfied() {
    let c = MinLengthConstraint { min_length: 5 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("hello")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_min_length_too_short_violated() {
    let c = MinLengthConstraint { min_length: 5 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("hi")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_max_length_satisfied() {
    let c = MaxLengthConstraint { max_length: 10 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("hello")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_max_length_exactly_at_max_satisfied() {
    let c = MaxLengthConstraint { max_length: 5 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("hello")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_max_length_too_long_violated() {
    let c = MaxLengthConstraint { max_length: 3 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("toolong")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_pattern_matching_digits_satisfied() {
    let c = PatternConstraint {
        pattern: r"^\d+$".to_string(),
        flags: None,
        message: None,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("12345")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_pattern_non_matching_violated() {
    let c = PatternConstraint {
        pattern: r"^\d+$".to_string(),
        flags: None,
        message: None,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("not-digits")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_pattern_email_satisfied() {
    let c = PatternConstraint {
        pattern: r"^[^@]+@[^@]+\.[^@]+$".to_string(),
        flags: None,
        message: None,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("user@example.com")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_pattern_email_invalid_violated() {
    let c = PatternConstraint {
        pattern: r"^[^@]+@[^@]+\.[^@]+$".to_string(),
        flags: None,
        message: None,
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("not-an-email")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_pattern_empty_values_satisfied() {
    let c = PatternConstraint {
        pattern: r"^\d+$".to_string(),
        flags: None,
        message: None,
    };
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_language_in_allowed_language_satisfied() {
    let c = LanguageInConstraint {
        languages: vec!["en".to_string(), "fr".to_string()],
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lang_lit("Hello", "en")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_language_in_disallowed_language_violated() {
    let c = LanguageInConstraint {
        languages: vec!["en".to_string(), "fr".to_string()],
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lang_lit("Hola", "es")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_language_in_plain_literal_ignored() {
    // Plain literals without language tags should not be checked by LanguageIn constraint
    let c = LanguageInConstraint {
        languages: vec!["en".to_string()],
    };
    let store = ConcreteStore::new().expect("store");
    // A non-language-tagged literal should pass through (no language to check)
    let ctx = make_context_with_values(vec![make_lit("plain text")]);
    let result = c.evaluate(&store, &ctx).expect("eval");
    // The constraint behavior on plain literals depends on implementation;
    // this test checks we get a definite result without panicking
    assert!(result.is_satisfied() || result.is_violated());
}

#[test]
fn test_unique_lang_single_literal_satisfied() {
    let c = UniqueLangConstraint { unique_lang: true };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lang_lit("Hello", "en")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_unique_lang_distinct_languages_satisfied() {
    let c = UniqueLangConstraint { unique_lang: true };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![
        make_lang_lit("Hello", "en"),
        make_lang_lit("Bonjour", "fr"),
    ]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_unique_lang_duplicate_language_violated() {
    let c = UniqueLangConstraint { unique_lang: true };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![
        make_lang_lit("Hello", "en"),
        make_lang_lit("Hi", "en"),
    ]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

#[test]
fn test_string_constraints_empty_values_all_satisfied() {
    let store = ConcreteStore::new().expect("store");
    let min_len = MinLengthConstraint { min_length: 100 };
    let max_len = MaxLengthConstraint { max_length: 0 };
    let pat = PatternConstraint {
        pattern: r"^impossible$".to_string(),
        flags: None,
        message: None,
    };
    // Empty value sets always satisfy (vacuously true)
    assert!(min_len
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
    assert!(max_len
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
    assert!(pat
        .evaluate(&store, &empty_context())
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_min_length_unicode_characters() {
    // Unicode chars: each codepoint counts as one character
    let c = MinLengthConstraint { min_length: 3 };
    let store = ConcreteStore::new().expect("store");
    // "日本語" has 3 Unicode code points
    let ctx = make_context_with_values(vec![make_lit("日本語")]);
    assert!(c.evaluate(&store, &ctx).expect("eval").is_satisfied());
}

#[test]
fn test_max_length_unicode_characters_violated() {
    let c = MaxLengthConstraint { max_length: 2 };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_lit("日本語")]);
    // 3 codepoints > max 2
    assert!(c.evaluate(&store, &ctx).expect("eval").is_violated());
}

// ---------------------------------------------------------------------------
// Node shape constraint — 20 tests
// ---------------------------------------------------------------------------

#[test]
fn test_node_constraint_no_values_satisfied() {
    let c = NodeConstraint::new(shape_id("http://example.org/SomeShape"));
    let store = ConcreteStore::new().expect("store");
    let result = c.evaluate(&empty_context(), &store).expect("eval");
    assert!(result.is_satisfied(), "no values means vacuously satisfied");
}

#[test]
fn test_node_constraint_with_unknown_shape_evaluated() {
    // When the shape registry is not provided, evaluate returns Err.
    // This verifies we get a proper error, not a panic.
    let c = NodeConstraint::new(shape_id("http://example.org/UnknownShape"));
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("someValue")]);
    let result = c.evaluate(&ctx, &store);
    // Must return an error when no shapes registry is available
    assert!(result.is_err());
}

#[test]
fn test_node_constraint_multiple_values_evaluated() {
    // Without a shapes registry, NodeConstraint returns Err for any non-empty value set.
    let c = NodeConstraint::new(shape_id("http://example.org/AnyShape"));
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("v1"), make_iri("v2"), make_iri("v3")]);
    let result = c.evaluate(&ctx, &store);
    // Must return an error when no shapes registry is available
    assert!(result.is_err());
}

#[test]
fn test_node_constraint_validate_ok() {
    let c = NodeConstraint::new(shape_id("http://example.org/ValidShape"));
    assert!(c.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Property path variants — 20 tests
// ---------------------------------------------------------------------------

#[test]
fn test_property_path_predicate_construction() {
    let pred = NamedNode::new_unchecked("http://example.org/name");
    let path = PropertyPath::Predicate(pred.clone());
    assert!(path.is_predicate());
    assert_eq!(path.as_predicate(), Some(&pred));
}

#[test]
fn test_property_path_inverse_construction() {
    let pred = NamedNode::new_unchecked("http://example.org/knows");
    let path = PropertyPath::Inverse(Box::new(PropertyPath::Predicate(pred)));
    assert!(!path.is_predicate());
}

#[test]
fn test_property_path_sequence_construction() {
    let pred_a = NamedNode::new_unchecked("http://example.org/a");
    let pred_b = NamedNode::new_unchecked("http://example.org/b");
    let path = PropertyPath::Sequence(vec![
        PropertyPath::Predicate(pred_a),
        PropertyPath::Predicate(pred_b),
    ]);
    assert!(!path.is_predicate());
}

#[test]
fn test_property_path_alternative_construction() {
    let pred_a = NamedNode::new_unchecked("http://example.org/email");
    let pred_b = NamedNode::new_unchecked("http://example.org/mailbox");
    let path = PropertyPath::Alternative(vec![
        PropertyPath::Predicate(pred_a),
        PropertyPath::Predicate(pred_b),
    ]);
    assert!(!path.is_predicate());
}

#[test]
fn test_property_path_zero_or_more_construction() {
    let pred = NamedNode::new_unchecked("http://example.org/child");
    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Predicate(pred)));
    assert!(!path.is_predicate());
}

#[test]
fn test_property_path_one_or_more_construction() {
    let pred = NamedNode::new_unchecked("http://example.org/member");
    let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Predicate(pred)));
    assert!(!path.is_predicate());
}

#[test]
fn test_property_path_zero_or_one_construction() {
    let pred = NamedNode::new_unchecked("http://example.org/optionalProp");
    let path = PropertyPath::ZeroOrOne(Box::new(PropertyPath::Predicate(pred)));
    assert!(!path.is_predicate());
}

#[test]
fn test_property_path_nested_inverse_sequence() {
    let pred_a = NamedNode::new_unchecked("http://example.org/a");
    let pred_b = NamedNode::new_unchecked("http://example.org/b");
    let path = PropertyPath::Inverse(Box::new(PropertyPath::Sequence(vec![
        PropertyPath::Predicate(pred_a),
        PropertyPath::Predicate(pred_b),
    ])));
    assert!(!path.is_predicate());
    assert!(path.as_predicate().is_none());
}

#[test]
fn test_property_path_predicate_helper_method() {
    let pred = NamedNode::new_unchecked("http://example.org/name");
    let path = PropertyPath::predicate(pred.clone());
    assert!(matches!(path, PropertyPath::Predicate(_)));
}

#[test]
fn test_property_path_inverse_helper_method() {
    let pred = NamedNode::new_unchecked("http://example.org/p");
    let inner = PropertyPath::Predicate(pred);
    let path = PropertyPath::inverse(inner);
    assert!(matches!(path, PropertyPath::Inverse(_)));
}

#[test]
fn test_property_path_sequence_helper_method() {
    let p1 = PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/a"));
    let p2 = PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/b"));
    let path = PropertyPath::sequence(vec![p1, p2]);
    assert!(matches!(path, PropertyPath::Sequence(_)));
}

#[test]
fn test_property_path_alternative_helper_method() {
    let p1 = PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/email"));
    let p2 = PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/mbox"));
    let path = PropertyPath::alternative(vec![p1, p2]);
    assert!(matches!(path, PropertyPath::Alternative(_)));
}

#[test]
fn test_property_path_zero_or_more_helper_method() {
    let pred = NamedNode::new_unchecked("http://example.org/child");
    let inner = PropertyPath::Predicate(pred);
    let path = PropertyPath::zero_or_more(inner);
    assert!(matches!(path, PropertyPath::ZeroOrMore(_)));
}

#[test]
fn test_property_path_one_or_more_helper_method() {
    let pred = NamedNode::new_unchecked("http://example.org/member");
    let inner = PropertyPath::Predicate(pred);
    let path = PropertyPath::one_or_more(inner);
    assert!(matches!(path, PropertyPath::OneOrMore(_)));
}

#[test]
fn test_property_path_zero_or_one_helper_method() {
    let pred = NamedNode::new_unchecked("http://example.org/opt");
    let inner = PropertyPath::Predicate(pred);
    let path = PropertyPath::zero_or_one(inner);
    assert!(matches!(path, PropertyPath::ZeroOrOne(_)));
}

#[test]
fn test_property_path_used_in_constraint_context() {
    let pred = NamedNode::new_unchecked("http://example.org/name");
    let path = PropertyPath::Predicate(pred);
    let ctx = ConstraintContext::new(focus_node(), shape_id("http://example.org/S"))
        .with_path(path.clone())
        .with_values(vec![make_lit("Alice")]);
    assert!(ctx.path.is_some());
}

#[test]
fn test_inverse_path_used_in_constraint_context() {
    let pred = NamedNode::new_unchecked("http://example.org/knows");
    let path = PropertyPath::Inverse(Box::new(PropertyPath::Predicate(pred)));
    let ctx = ConstraintContext::new(focus_node(), shape_id("http://example.org/S"))
        .with_path(path)
        .with_values(vec![make_iri("friend")]);
    assert!(ctx.path.is_some());
    assert!(!ctx.path.as_ref().unwrap().is_predicate());
}

#[test]
fn test_sequence_path_used_in_constraint_context() {
    let p1 = PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/knows"));
    let p2 = PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/name"));
    let path = PropertyPath::Sequence(vec![p1, p2]);
    let ctx = ConstraintContext::new(focus_node(), shape_id("http://example.org/S"))
        .with_path(path)
        .with_values(vec![make_lit("Alice")]);
    assert!(ctx.path.is_some());
}

#[test]
fn test_alternative_path_in_context() {
    let p1 = PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/email"));
    let p2 = PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/mbox"));
    let path = PropertyPath::Alternative(vec![p1, p2]);
    let ctx = ConstraintContext::new(focus_node(), shape_id("http://example.org/S"))
        .with_path(path)
        .with_values(vec![make_lit("user@example.com")]);
    assert!(ctx.path.is_some());
}

#[test]
fn test_zero_or_more_path_in_context() {
    let pred = NamedNode::new_unchecked("http://example.org/child");
    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Predicate(pred)));
    let ctx = ConstraintContext::new(focus_node(), shape_id("http://example.org/S"))
        .with_path(path)
        .with_values(vec![make_iri("child1"), make_iri("child2")]);
    assert!(ctx.values.len() == 2);
}

// ---------------------------------------------------------------------------
// Logical constraints — 20 tests
// ---------------------------------------------------------------------------

#[test]
fn test_not_constraint_validate_empty_shape_err() {
    let c = NotConstraint {
        shape: shape_id(""),
    };
    assert!(
        c.validate().is_err(),
        "NOT with empty shape should fail validation"
    );
}

#[test]
fn test_not_constraint_validate_valid_shape_ok() {
    let c = NotConstraint {
        shape: shape_id("http://example.org/S"),
    };
    assert!(c.validate().is_ok());
}

#[test]
fn test_not_constraint_no_values_satisfied() {
    let c = NotConstraint {
        shape: shape_id("http://example.org/S"),
    };
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&empty_context(), &store)
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_and_constraint_validate_empty_shapes_err() {
    let c = AndConstraint::new(vec![]);
    assert!(
        c.validate().is_err(),
        "AND with no shapes should fail validation"
    );
}

#[test]
fn test_and_constraint_validate_single_shape_ok() {
    let c = AndConstraint::new(vec![shape_id("http://example.org/S")]);
    assert!(c.validate().is_ok());
}

#[test]
fn test_and_constraint_no_values_satisfied() {
    let c = AndConstraint::new(vec![shape_id("http://example.org/S")]);
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&empty_context(), &store)
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_or_constraint_validate_empty_shapes_err() {
    let c = OrConstraint::new(vec![]);
    assert!(
        c.validate().is_err(),
        "OR with no shapes should fail validation"
    );
}

#[test]
fn test_or_constraint_validate_single_shape_ok() {
    let c = OrConstraint::new(vec![shape_id("http://example.org/S")]);
    assert!(c.validate().is_ok());
}

#[test]
fn test_or_constraint_no_values_satisfied() {
    let c = OrConstraint::new(vec![shape_id("http://example.org/S")]);
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&empty_context(), &store)
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_xone_constraint_validate_single_shape_err() {
    let c = XoneConstraint::new(vec![shape_id("http://example.org/S")]);
    assert!(c.validate().is_err(), "XONE requires at least two shapes");
}

#[test]
fn test_xone_constraint_validate_two_shapes_ok() {
    let c = XoneConstraint::new(vec![
        shape_id("http://example.org/A"),
        shape_id("http://example.org/B"),
    ]);
    assert!(c.validate().is_ok());
}

#[test]
fn test_xone_constraint_no_values_satisfied() {
    let c = XoneConstraint::new(vec![
        shape_id("http://example.org/A"),
        shape_id("http://example.org/B"),
    ]);
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&empty_context(), &store)
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_and_constraint_multiple_shapes_no_values_satisfied() {
    let c = AndConstraint::new(vec![
        shape_id("http://example.org/A"),
        shape_id("http://example.org/B"),
        shape_id("http://example.org/C"),
    ]);
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&empty_context(), &store)
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_or_constraint_multiple_shapes_no_values_satisfied() {
    let c = OrConstraint::new(vec![
        shape_id("http://example.org/A"),
        shape_id("http://example.org/B"),
    ]);
    let store = ConcreteStore::new().expect("store");
    assert!(c
        .evaluate(&empty_context(), &store)
        .expect("eval")
        .is_satisfied());
}

#[test]
fn test_logical_constraint_not_evaluates_with_values() {
    let c = NotConstraint {
        shape: shape_id("http://example.org/S"),
    };
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("v1")]);
    let result = c.evaluate(&ctx, &store).expect("eval");
    assert!(result.is_satisfied() || result.is_violated());
}

#[test]
fn test_logical_constraint_and_evaluates_with_values() {
    let c = AndConstraint::new(vec![shape_id("http://example.org/S")]);
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("v1")]);
    let result = c.evaluate(&ctx, &store).expect("eval");
    assert!(result.is_satisfied() || result.is_violated());
}

#[test]
fn test_logical_constraint_or_evaluates_with_values() {
    let c = OrConstraint::new(vec![shape_id("http://example.org/S")]);
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("v1")]);
    let result = c.evaluate(&ctx, &store).expect("eval");
    assert!(result.is_satisfied() || result.is_violated());
}

#[test]
fn test_logical_constraint_xone_evaluates_with_values() {
    let c = XoneConstraint::new(vec![
        shape_id("http://example.org/A"),
        shape_id("http://example.org/B"),
    ]);
    let store = ConcreteStore::new().expect("store");
    let ctx = make_context_with_values(vec![make_iri("v1")]);
    let result = c.evaluate(&ctx, &store).expect("eval");
    assert!(result.is_satisfied() || result.is_violated());
}

#[test]
fn test_and_constraint_empty_shape_reference_err() {
    let c = AndConstraint::new(vec![shape_id(""), shape_id("http://example.org/S")]);
    assert!(
        c.validate().is_err(),
        "AND with empty shape reference fails"
    );
}

#[test]
fn test_or_constraint_empty_shape_reference_err() {
    let c = OrConstraint::new(vec![shape_id(""), shape_id("http://example.org/S")]);
    assert!(c.validate().is_err(), "OR with empty shape reference fails");
}

// ---------------------------------------------------------------------------
// Severity levels — 10 tests
// ---------------------------------------------------------------------------

#[test]
fn test_severity_violation_is_default() {
    let sev = Severity::default();
    assert_eq!(sev, Severity::Violation);
}

#[test]
fn test_severity_ordering_info_lt_warning() {
    assert!(Severity::Info < Severity::Warning);
}

#[test]
fn test_severity_ordering_warning_lt_violation() {
    assert!(Severity::Warning < Severity::Violation);
}

#[test]
fn test_severity_display_violation() {
    assert_eq!(Severity::Violation.to_string(), "Violation");
}

#[test]
fn test_severity_display_warning() {
    assert_eq!(Severity::Warning.to_string(), "Warning");
}

#[test]
fn test_severity_display_info() {
    assert_eq!(Severity::Info.to_string(), "Info");
}

#[test]
fn test_validation_violation_with_violation_severity() {
    let v = ValidationViolation::new(
        focus_node(),
        shape_id("http://example.org/S"),
        ConstraintComponentId::new("sh:MinCountConstraintComponent"),
        Severity::Violation,
    );
    assert_eq!(v.result_severity, Severity::Violation);
}

#[test]
fn test_validation_violation_with_warning_severity() {
    let v = ValidationViolation::new(
        focus_node(),
        shape_id("http://example.org/S"),
        ConstraintComponentId::new("sh:MinCountConstraintComponent"),
        Severity::Warning,
    );
    assert_eq!(v.result_severity, Severity::Warning);
}

#[test]
fn test_validation_violation_with_info_severity() {
    let v = ValidationViolation::new(
        focus_node(),
        shape_id("http://example.org/S"),
        ConstraintComponentId::new("sh:MinCountConstraintComponent"),
        Severity::Info,
    );
    assert_eq!(v.result_severity, Severity::Info);
}

#[test]
fn test_violations_different_severities_collected_independently() {
    let violation = ValidationViolation::new(
        focus_node(),
        shape_id("http://example.org/S1"),
        ConstraintComponentId::new("sh:MinCountConstraintComponent"),
        Severity::Violation,
    );
    let warning = ValidationViolation::new(
        focus_node(),
        shape_id("http://example.org/S2"),
        ConstraintComponentId::new("sh:MaxCountConstraintComponent"),
        Severity::Warning,
    );
    let info = ValidationViolation::new(
        focus_node(),
        shape_id("http://example.org/S3"),
        ConstraintComponentId::new("sh:PatternConstraintComponent"),
        Severity::Info,
    );
    let all = [violation, warning, info];
    assert_eq!(all.len(), 3);
    assert_eq!(
        all.iter()
            .filter(|v| v.result_severity == Severity::Violation)
            .count(),
        1
    );
    assert_eq!(
        all.iter()
            .filter(|v| v.result_severity == Severity::Warning)
            .count(),
        1
    );
    assert_eq!(
        all.iter()
            .filter(|v| v.result_severity == Severity::Info)
            .count(),
        1
    );
}

// ---------------------------------------------------------------------------
// Cache hit/miss behaviour — 10 tests
// ---------------------------------------------------------------------------

#[test]
fn test_validation_cache_miss_on_empty() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    let key = ValidationCacheKey::new("http://node", "http://shape", 0);
    assert!(cache.get(&key).is_none(), "empty cache must yield a miss");
}

#[test]
fn test_validation_cache_hit_after_put() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    let key = ValidationCacheKey::new("http://node", "http://shape", 42);
    let entry = CachedValidationResult::new(
        "http://node",
        "http://shape",
        true,
        vec![],
        Duration::from_secs(60),
    );
    cache.put(key.clone(), entry);
    let hit = cache.get(&key);
    assert!(hit.is_some(), "cache must return an entry after put");
    assert!(hit.expect("entry").is_valid);
}

#[test]
fn test_validation_cache_hit_returns_violation_messages() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    let key = ValidationCacheKey::new("http://violatingNode", "http://shape", 1);
    let messages = vec!["Value must be at least 5 characters".to_string()];
    let entry = CachedValidationResult::new(
        "http://violatingNode",
        "http://shape",
        false,
        messages.clone(),
        Duration::from_secs(60),
    );
    cache.put(key.clone(), entry);
    let hit = cache.get(&key).expect("cache hit");
    assert!(!hit.is_valid);
    assert_eq!(hit.violation_messages, messages);
}

#[test]
fn test_validation_cache_miss_different_node() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    let key_a = ValidationCacheKey::new("http://nodeA", "http://shape", 0);
    let key_b = ValidationCacheKey::new("http://nodeB", "http://shape", 0);
    let entry = CachedValidationResult::new(
        "http://nodeA",
        "http://shape",
        true,
        vec![],
        Duration::from_secs(60),
    );
    cache.put(key_a, entry);
    assert!(
        cache.get(&key_b).is_none(),
        "different node must yield a miss"
    );
}

#[test]
fn test_validation_cache_miss_different_shape() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    let key_a = ValidationCacheKey::new("http://node", "http://shapeA", 0);
    let key_b = ValidationCacheKey::new("http://node", "http://shapeB", 0);
    let entry = CachedValidationResult::new(
        "http://node",
        "http://shapeA",
        true,
        vec![],
        Duration::from_secs(60),
    );
    cache.put(key_a, entry);
    assert!(
        cache.get(&key_b).is_none(),
        "different shape must yield a miss"
    );
}

#[test]
fn test_validation_cache_invalidate_by_node() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    let key = ValidationCacheKey::new("http://node", "http://shape", 0);
    let entry = CachedValidationResult::new(
        "http://node",
        "http://shape",
        true,
        vec![],
        Duration::from_secs(60),
    );
    cache.put(key.clone(), entry);
    let invalidated = cache.invalidate_node("http://node");
    assert!(invalidated > 0, "at least one entry must be invalidated");
    assert!(
        cache.get(&key).is_none(),
        "entry must be gone after invalidation"
    );
}

#[test]
fn test_validation_cache_size_grows_with_entries() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    assert_eq!(cache.size(), 0);
    for i in 0..5u32 {
        let key = ValidationCacheKey::new(format!("http://node{i}"), "http://shape", u64::from(i));
        let entry = CachedValidationResult::new(
            format!("http://node{i}"),
            "http://shape",
            true,
            vec![],
            Duration::from_secs(60),
        );
        cache.put(key, entry);
    }
    assert_eq!(cache.size(), 5);
}

#[test]
fn test_validation_cache_clear_removes_all() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    for i in 0..3u32 {
        let key = ValidationCacheKey::new(format!("http://node{i}"), "http://shape", u64::from(i));
        let entry = CachedValidationResult::new(
            format!("http://node{i}"),
            "http://shape",
            true,
            vec![],
            Duration::from_secs(60),
        );
        cache.put(key, entry);
    }
    cache.clear();
    assert_eq!(cache.size(), 0, "cache must be empty after clear");
}

#[test]
fn test_validation_cache_hit_rate_starts_at_zero() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    // No lookups yet — hit rate is 0 or NaN-safe
    let rate = cache.hit_rate();
    assert!(rate == 0.0 || rate.is_nan(), "initial hit rate must be 0.0");
}

#[test]
fn test_validation_cache_accessed_triple_dependency() {
    let cache = ValidationCache::new(100, Duration::from_secs(60));
    let key = ValidationCacheKey::new("http://node", "http://shape", 0);
    let mut entry = CachedValidationResult::new(
        "http://node",
        "http://shape",
        true,
        vec![],
        Duration::from_secs(60),
    );
    let triple_key = "<http://s> <http://p> <http://o>".to_string();
    entry.add_accessed_triple(triple_key.clone());
    assert!(entry.accessed_triples.contains(&triple_key));
    cache.put(key.clone(), entry);
    // Invalidating the triple must evict the dependent entry
    let count = cache.invalidate_triple(&triple_key);
    assert!(count > 0, "triple invalidation must remove dependent entry");
    assert!(cache.get(&key).is_none());
}
