//! Type System Conformance Tests
//!
//! Additional tests for the SPARQL 1.1 type system, RDF term types, and entailment.

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== IRI HANDLING =====

#[test]
fn test_ts_iri_01_exact_match() {
    // Exact IRI matching
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(
            ex("alice"),
            rdf_type(),
            iri(&format!("{FOAF}Person")),
        )]),
        vec![],
    );
    let test = ConformanceTest::new(
        "ts-iri-01",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("ts-iri-01 failed");
}

#[test]
fn test_ts_iri_02_iri_variable() {
    // Variable matches IRI
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), rdf_type(), var("t"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsIri,
                operand: Box::new(expr_var("t")),
            },
        ),
        vec![variable("t")],
    );
    let test = ConformanceTest::new(
        "ts-iri-02",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // 3 persons, type is IRI
    );
    runner().run_test(&test).expect("ts-iri-02 failed");
}

// ===== BLANK NODE HANDLING =====

#[test]
fn test_ts_bnode_01_detection() {
    // Detect blank nodes using isBlank
    let mut ds = InMemoryDataset::new();
    ds.add_triple(
        crate::algebra::Term::BlankNode("b1".to_string()),
        foaf("name"),
        str_lit("Anonymous"),
    );
    ds.add_triple(ex("known"), foaf("name"), str_lit("Known"));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsBlank,
                operand: Box::new(expr_var("s")),
            },
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "ts-bnode-01",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("ts-bnode-01 failed");
}

#[test]
fn test_ts_bnode_02_not_blank() {
    // NOT isBlank filter
    let mut ds = InMemoryDataset::new();
    ds.add_triple(
        crate::algebra::Term::BlankNode("b1".to_string()),
        foaf("name"),
        str_lit("Anonymous"),
    );
    ds.add_triple(ex("known"), foaf("name"), str_lit("Known"));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::Not,
                operand: Box::new(Expression::Unary {
                    op: crate::algebra::UnaryOperator::IsBlank,
                    operand: Box::new(expr_var("s")),
                }),
            },
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "ts-bnode-02",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only "Known"
    );
    runner().run_test(&test).expect("ts-bnode-02 failed");
}

// ===== TYPED LITERAL COMPARISONS =====

#[test]
fn test_ts_typed_01_integer_eq() {
    // Integer equality
    let ds = numeric_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            expr_eq(expr_var("v"), expr_lit(lit_int(20))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "ts-typed-01",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // s2 = 20
    );
    runner().run_test(&test).expect("ts-typed-01 failed");
}

#[test]
fn test_ts_typed_02_integer_range() {
    // Integer range
    let ds = numeric_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            expr_and(
                expr_gt(expr_var("v"), expr_lit(lit_int(10))),
                expr_lt(expr_var("v"), expr_lit(lit_int(30))),
            ),
        ),
        vec![variable("s"), variable("v")],
    );
    let test = ConformanceTest::new(
        "ts-typed-02",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // 15, 20, 25 => all 3 between 10 and 30 exclusive
    );
    runner().run_test(&test).expect("ts-typed-02 failed");
}

#[test]
fn test_ts_typed_03_string_comparison() {
    // String comparison
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: Box::new(expr_var("name")),
                right: Box::new(expr_lit(lit_str("B"))),
            },
        ),
        vec![variable("name")],
    );
    // String comparison: Bob >= B and Charlie >= B
    let test = ConformanceTest::new(
        "ts-typed-03",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // Bob, Charlie >= "B"
    );
    runner().run_test(&test).expect("ts-typed-03 failed");
}

// ===== SAME TERM SEMANTICS =====

#[test]
fn test_ts_sameterm_01() {
    // sameTerm distinguishes IRIs
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Binary {
                op: BinaryOperator::SameTerm,
                left: Box::new(expr_var("s")),
                right: Box::new(expr_iri("http://example.org/bob")),
            },
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "ts-sameterm-01",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("name", "Bob")])]),
    );
    runner().run_test(&test).expect("ts-sameterm-01 failed");
}

#[test]
fn test_ts_sameterm_02_literals() {
    // sameTerm on literals (exact match)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Binary {
                op: BinaryOperator::SameTerm,
                left: Box::new(expr_var("name")),
                right: Box::new(expr_lit(lit_str("Alice"))),
            },
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "ts-sameterm-02",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("ts-sameterm-02 failed");
}

// ===== BOUND / UNBOUND HANDLING =====

#[test]
fn test_ts_bound_01_all_bound() {
    // BOUND always true when variable in BGP result
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Bound(variable("name")),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "ts-bound-01",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // all names are bound
    );
    runner().run_test(&test).expect("ts-bound-01 failed");
}

#[test]
fn test_ts_bound_02_optional() {
    // BOUND on optional variable
    let ds = optional_dataset();
    let algebra = project(
        filter(
            left_join(
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
                bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
                None,
            ),
            Expression::Bound(variable("mbox")),
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "ts-bound-02",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // alice and charlie have mbox
    );
    runner().run_test(&test).expect("ts-bound-02 failed");
}

// ===== CONDITIONAL EXPRESSION =====

#[test]
fn test_ts_if_01_basic() {
    // IF(?age > 30, "old", "young") — simplified via conditional expression
    let ds = person_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            variable("category"),
            Expression::Conditional {
                condition: Box::new(expr_gt(expr_var("age"), expr_lit(lit_int(30)))),
                then_expr: Box::new(expr_lit(lit_str("old"))),
                else_expr: Box::new(expr_lit(lit_str("young"))),
            },
        ),
        vec![variable("s"), variable("category")],
    );
    let test = ConformanceTest::new(
        "ts-if-01",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("ts-if-01 failed");
}

// ===== ENTAILMENT TESTS (Simple RDF entailment) =====

#[test]
fn test_ent_01_reflexive_type() {
    // Simple entailment: :alice is a Person
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(
            ex("alice"),
            rdf_type(),
            iri(&format!("{FOAF}Person")),
        )]),
        vec![],
    );
    let test = ConformanceTest::new(
        "ent-01",
        ConformanceGroup::Entailment,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("ent-01 failed");
}

#[test]
fn test_ent_02_triple_existence() {
    // Verify specific triple exists
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(ex("alice"), foaf("name"), str_lit("Alice"))]),
        vec![],
    );
    let test = ConformanceTest::new(
        "ent-02",
        ConformanceGroup::Entailment,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("ent-02 failed");
}

#[test]
fn test_ent_03_nonexistent_triple() {
    // Verify non-existent triple returns empty
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(ex("alice"), foaf("name"), str_lit("Bob"))]),
        vec![],
    );
    let test = ConformanceTest::new(
        "ent-03",
        ConformanceGroup::Entailment,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("ent-03 failed");
}
