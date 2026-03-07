//! FILTER Expression Conformance Tests
//!
//! Tests SPARQL 1.1 FILTER expressions including comparison operators,
//! logical operators, type-checking functions, and built-in functions.

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== COMPARISON FILTER TESTS =====

#[test]
fn test_filter_eq_01_string() {
    // FILTER(?name = "Alice")
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            expr_eq(expr_var("name"), expr_lit(lit_str("Alice"))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "filter-eq-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("filter-eq-01 failed");
}

#[test]
fn test_filter_eq_02_iri() {
    // FILTER(?s = :alice)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            expr_eq(expr_var("s"), expr_iri("http://example.org/alice")),
        ),
        vec![variable("s"), variable("name")],
    );
    let test = ConformanceTest::new(
        "filter-eq-02",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("filter-eq-02 failed");
}

#[test]
fn test_filter_neq_01() {
    // FILTER(?name != "Alice")
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Binary {
                op: BinaryOperator::NotEqual,
                left: Box::new(expr_var("name")),
                right: Box::new(expr_lit(lit_str("Alice"))),
            },
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "filter-neq-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // Bob, Charlie
    );
    runner().run_test(&test).expect("filter-neq-01 failed");
}

#[test]
fn test_filter_lt_01() {
    // FILTER(?age < 30)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            expr_lt(expr_var("age"), expr_lit(lit_int(30))),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "filter-lt-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // Bob (25)
    );
    runner().run_test(&test).expect("filter-lt-01 failed");
}

#[test]
fn test_filter_gt_01() {
    // FILTER(?age > 30)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            expr_gt(expr_var("age"), expr_lit(lit_int(30))),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "filter-gt-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // Charlie (35)
    );
    runner().run_test(&test).expect("filter-gt-01 failed");
}

#[test]
fn test_filter_le_01() {
    // FILTER(?age <= 30)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            Expression::Binary {
                op: BinaryOperator::LessEqual,
                left: Box::new(expr_var("age")),
                right: Box::new(expr_lit(lit_int(30))),
            },
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "filter-le-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // Bob (25), Alice (30)
    );
    runner().run_test(&test).expect("filter-le-01 failed");
}

#[test]
fn test_filter_ge_01() {
    // FILTER(?age >= 30)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: Box::new(expr_var("age")),
                right: Box::new(expr_lit(lit_int(30))),
            },
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "filter-ge-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // Alice (30), Charlie (35)
    );
    runner().run_test(&test).expect("filter-ge-01 failed");
}

// ===== LOGICAL FILTER TESTS =====

#[test]
fn test_filter_and_01() {
    // FILTER(?age > 25 && ?age < 35)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            expr_and(
                expr_gt(expr_var("age"), expr_lit(lit_int(25))),
                expr_lt(expr_var("age"), expr_lit(lit_int(35))),
            ),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "filter-and-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only Alice (30)
    );
    runner().run_test(&test).expect("filter-and-01 failed");
}

#[test]
fn test_filter_or_01() {
    // FILTER(?age < 26 || ?age > 34)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            expr_or(
                expr_lt(expr_var("age"), expr_lit(lit_int(26))),
                expr_gt(expr_var("age"), expr_lit(lit_int(34))),
            ),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "filter-or-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // Bob (25), Charlie (35)
    );
    runner().run_test(&test).expect("filter-or-01 failed");
}

#[test]
fn test_filter_not_01() {
    // FILTER(!(?age = 30))
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            Expression::Unary {
                op: UnaryOperator::Not,
                operand: Box::new(expr_eq(expr_var("age"), expr_lit(lit_int(30)))),
            },
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "filter-not-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // Bob (25), Charlie (35)
    );
    runner().run_test(&test).expect("filter-not-01 failed");
}

// ===== TYPE CHECKING FILTER TESTS =====

#[test]
fn test_filter_isiri_01() {
    // FILTER(isIRI(?s))
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Unary {
                op: UnaryOperator::IsIri,
                operand: Box::new(expr_var("s")),
            },
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "filter-isiri-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // all subjects are IRIs
    );
    runner().run_test(&test).expect("filter-isiri-01 failed");
}

#[test]
fn test_filter_isliteral_01() {
    // FILTER(isLiteral(?name))
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Unary {
                op: UnaryOperator::IsLiteral,
                operand: Box::new(expr_var("name")),
            },
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "filter-isliteral-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // all names are literals
    );
    runner()
        .run_test(&test)
        .expect("filter-isliteral-01 failed");
}

#[test]
fn test_filter_isblank_01() {
    // FILTER(isBlank(?s)) — no blank nodes in our test data
    let mut ds = InMemoryDataset::new();
    ds.add_triple(
        crate::algebra::Term::BlankNode("b1".to_string()),
        foaf("name"),
        str_lit("BlankPerson"),
    );
    ds.add_triple(ex("a"), foaf("name"), str_lit("IRIPerson"));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Unary {
                op: UnaryOperator::IsBlank,
                operand: Box::new(expr_var("s")),
            },
        ),
        vec![variable("s"), variable("name")],
    );
    let test = ConformanceTest::new(
        "filter-isblank-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only blank node subject
    );
    runner().run_test(&test).expect("filter-isblank-01 failed");
}

#[test]
fn test_filter_isnumeric_01() {
    // FILTER(isNumeric(?v))
    let ds = numeric_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            Expression::Unary {
                op: UnaryOperator::IsNumeric,
                operand: Box::new(expr_var("v")),
            },
        ),
        vec![variable("v")],
    );
    let test = ConformanceTest::new(
        "filter-isnumeric-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5), // all values are numeric
    );
    runner()
        .run_test(&test)
        .expect("filter-isnumeric-01 failed");
}

// ===== SAMETERM FILTER =====

#[test]
fn test_filter_sameterm_01() {
    // FILTER(sameTerm(?s, :alice))
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Binary {
                op: BinaryOperator::SameTerm,
                left: Box::new(expr_var("s")),
                right: Box::new(expr_iri("http://example.org/alice")),
            },
        ),
        vec![variable("s"), variable("name")],
    );
    let test = ConformanceTest::new(
        "filter-sameterm-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("filter-sameterm-01 failed");
}

// ===== BOUND FILTER =====

#[test]
fn test_filter_bound_01() {
    // FILTER(BOUND(?mbox))
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
        vec![variable("s"), variable("name")],
    );
    let test = ConformanceTest::new(
        "filter-bound-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // alice and charlie have mbox
    );
    runner().run_test(&test).expect("filter-bound-01 failed");
}

// ===== STR FUNCTION FILTER =====

#[test]
fn test_filter_str_01() {
    // FILTER(str(?s) = "http://example.org/alice")
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            expr_eq(
                expr_fn("str", vec![expr_var("s")]),
                expr_lit(lit_str("http://example.org/alice")),
            ),
        ),
        vec![variable("s"), variable("name")],
    );
    let test = ConformanceTest::new(
        "filter-str-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("filter-str-01 failed");
}

// ===== LANG FUNCTION FILTER =====

#[test]
fn test_filter_lang_01() {
    // FILTER(LANG(?label) = "en")
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("r1"), ex("label"), lang_lit("Hello", "en"));
    ds.add_triple(ex("r2"), ex("label"), lang_lit("Bonjour", "fr"));
    ds.add_triple(ex("r3"), ex("label"), lang_lit("Hello", "en"));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("label"), var("label"))]),
            expr_eq(
                expr_fn("lang", vec![expr_var("label")]),
                expr_lit(lit_str("en")),
            ),
        ),
        vec![variable("s"), variable("label")],
    );
    let test = ConformanceTest::new(
        "filter-lang-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // r1 and r3
    );
    runner().run_test(&test).expect("filter-lang-01 failed");
}

// ===== COMPLEX FILTER TESTS =====

#[test]
fn test_filter_complex_01_nested_and_or() {
    // FILTER((?age > 25 && ?age < 30) || ?age > 33)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            expr_or(
                expr_and(
                    expr_gt(expr_var("age"), expr_lit(lit_int(25))),
                    expr_lt(expr_var("age"), expr_lit(lit_int(30))),
                ),
                expr_gt(expr_var("age"), expr_lit(lit_int(33))),
            ),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "filter-complex-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // Charlie only (35 > 33); bob=25 not > 25
    );
    runner().run_test(&test).expect("filter-complex-01 failed");
}

#[test]
fn test_filter_rejects_all_01() {
    // FILTER that rejects everything
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            expr_eq(expr_var("name"), expr_lit(lit_str("Nobody"))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "filter-reject-all-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner()
        .run_test(&test)
        .expect("filter-reject-all-01 failed");
}

#[test]
fn test_filter_accepts_all_01() {
    // FILTER that accepts everything (trivially true)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            expr_eq(expr_lit(lit_int(1)), expr_lit(lit_int(1))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "filter-accept-all-01",
        ConformanceGroup::FilterExpressions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner()
        .run_test(&test)
        .expect("filter-accept-all-01 failed");
}
