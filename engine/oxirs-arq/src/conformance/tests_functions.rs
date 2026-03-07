//! String, Date, and Math Function Conformance Tests
//!
//! Tests SPARQL 1.1 built-in function expressions for strings, dates, and math.

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== STRING FUNCTION TESTS =====

#[test]
fn test_str_fn_01_str_of_iri() {
    // SELECT (str(:alice) AS ?s) WHERE {}
    let ds = InMemoryDataset::new();
    let algebra = project(
        extend(
            bgp(vec![]),
            variable("result"),
            expr_fn("str", vec![expr_iri("http://example.org/alice")]),
        ),
        vec![variable("result")],
    );
    let test = ConformanceTest::new(
        "str-fn-01",
        ConformanceGroup::StringFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(0), // empty BGP = no results
    );
    runner().run_test(&test).expect("str-fn-01 failed");
}

#[test]
fn test_str_fn_02_str_filter() {
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
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "str-fn-02",
        ConformanceGroup::StringFunctions,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("name", "Alice")])]),
    );
    runner().run_test(&test).expect("str-fn-02 failed");
}

#[test]
fn test_str_fn_03_lang_filter() {
    // FILTER(lang(?label) = "en")
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("r1"), ex("label"), lang_lit("Hello", "en"));
    ds.add_triple(ex("r2"), ex("label"), lang_lit("Bonjour", "fr"));
    ds.add_triple(ex("r3"), ex("label"), lang_lit("Hi", "en"));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("label"), var("label"))]),
            expr_eq(
                expr_fn("lang", vec![expr_var("label")]),
                expr_lit(lit_str("en")),
            ),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "str-fn-03",
        ConformanceGroup::StringFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // r1 and r3
    );
    runner().run_test(&test).expect("str-fn-03 failed");
}

#[test]
fn test_str_fn_04_datatype_filter() {
    // FILTER(datatype(?v) = xsd:integer)
    let ds = numeric_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            expr_eq(
                expr_fn("datatype", vec![expr_var("v")]),
                expr_iri("http://www.w3.org/2001/XMLSchema#integer"),
            ),
        ),
        vec![variable("s"), variable("v")],
    );
    let test = ConformanceTest::new(
        "str-fn-04",
        ConformanceGroup::StringFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5), // all values are integers
    );
    runner().run_test(&test).expect("str-fn-04 failed");
}

#[test]
fn test_str_fn_05_isiri_check() {
    // FILTER(isIRI(?s))
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsIri,
                operand: Box::new(expr_var("s")),
            },
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "str-fn-05",
        ConformanceGroup::StringFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("str-fn-05 failed");
}

#[test]
fn test_str_fn_06_isliteral_check() {
    // FILTER(isLiteral(?name))
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsLiteral,
                operand: Box::new(expr_var("name")),
            },
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "str-fn-06",
        ConformanceGroup::StringFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("str-fn-06 failed");
}

#[test]
fn test_str_fn_07_isblank_check() {
    // FILTER(isBlank(?bn)) — tests with explicit blank node data
    let mut ds = InMemoryDataset::new();
    ds.add_triple(
        crate::algebra::Term::BlankNode("b1".to_string()),
        ex("p"),
        str_lit("blank subject"),
    );
    ds.add_triple(ex("iri"), ex("p"), str_lit("iri subject"));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsBlank,
                operand: Box::new(expr_var("s")),
            },
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "str-fn-07",
        ConformanceGroup::StringFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("str-fn-07 failed");
}

#[test]
fn test_str_fn_08_isnumeric_check() {
    // FILTER(isNumeric(?v))
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("r1"), ex("val"), int_lit(42));
    ds.add_triple(ex("r2"), ex("val"), str_lit("hello"));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsNumeric,
                operand: Box::new(expr_var("v")),
            },
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "str-fn-08",
        ConformanceGroup::StringFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only r1
    );
    runner().run_test(&test).expect("str-fn-08 failed");
}

// ===== MATH FUNCTION TESTS =====

#[test]
fn test_math_fn_01_add() {
    // FILTER(10 + 5 = 15)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            expr_eq(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: Box::new(expr_lit(lit_int(10))),
                    right: Box::new(expr_lit(lit_int(5))),
                },
                expr_lit(lit_int(15)),
            ),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "math-fn-01",
        ConformanceGroup::MathFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // filter is always true (10+5=15)
    );
    runner().run_test(&test).expect("math-fn-01 failed");
}

#[test]
fn test_math_fn_02_subtract() {
    // FILTER(?age - 5 > 25) => age > 30 => only charlie (35)
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            expr_gt(
                Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: Box::new(expr_var("age")),
                    right: Box::new(expr_lit(lit_int(5))),
                },
                expr_lit(lit_int(25)),
            ),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "math-fn-02",
        ConformanceGroup::MathFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // alice(30-5=25, not > 25), bob(25-5=20, not > 25), charlie(35-5=30>25) => only charlie
    );
    runner().run_test(&test).expect("math-fn-02 failed");
}

#[test]
fn test_math_fn_03_multiply() {
    // FILTER(?v * 2 > 30) where values are 10,15,20,25,30
    // 2*10=20, 2*15=30, 2*20=40>30, 2*25=50>30, 2*30=60>30 => 3 results
    let ds = numeric_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            expr_gt(
                Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: Box::new(expr_var("v")),
                    right: Box::new(expr_lit(lit_int(2))),
                },
                expr_lit(lit_int(30)),
            ),
        ),
        vec![variable("s"), variable("v")],
    );
    let test = ConformanceTest::new(
        "math-fn-03",
        ConformanceGroup::MathFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // 20, 25, 30 each * 2 > 30
    );
    runner().run_test(&test).expect("math-fn-03 failed");
}

#[test]
fn test_math_fn_04_divide() {
    // FILTER(?v / 5 > 3) where values 10,15,20,25,30
    // 10/5=2, 15/5=3, 20/5=4>3, 25/5=5>3, 30/5=6>3 => 3 results
    let ds = numeric_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            expr_gt(
                Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: Box::new(expr_var("v")),
                    right: Box::new(expr_lit(lit_int(5))),
                },
                expr_lit(lit_int(3)),
            ),
        ),
        vec![variable("s"), variable("v")],
    );
    let test = ConformanceTest::new(
        "math-fn-04",
        ConformanceGroup::MathFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("math-fn-04 failed");
}

#[test]
fn test_math_fn_05_comparison_chain() {
    // Multiple numeric comparisons: ?age > 20 && ?age < 36
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            expr_and(
                expr_gt(expr_var("age"), expr_lit(lit_int(20))),
                expr_lt(expr_var("age"), expr_lit(lit_int(36))),
            ),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "math-fn-05",
        ConformanceGroup::MathFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // alice(30), bob(25), charlie(35) — all in range
    );
    runner().run_test(&test).expect("math-fn-05 failed");
}

// ===== TYPE SYSTEM TESTS =====

#[test]
fn test_type_01_integer_literal() {
    // Verify integer literal is properly typed
    let ds = typed_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("r"), ex("intVal"), var("v"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsNumeric,
                operand: Box::new(expr_var("v")),
            },
        ),
        vec![variable("r"), variable("v")],
    );
    let test = ConformanceTest::new(
        "type-01",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // r1 and r2 have intVal
    );
    runner().run_test(&test).expect("type-01 failed");
}

#[test]
fn test_type_02_string_literal() {
    // Verify string literals are typed
    let ds = typed_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("r"), ex("strVal"), var("v"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsLiteral,
                operand: Box::new(expr_var("v")),
            },
        ),
        vec![variable("r"), variable("v")],
    );
    let test = ConformanceTest::new(
        "type-02",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // r1 has strVal
    );
    runner().run_test(&test).expect("type-02 failed");
}

#[test]
fn test_type_03_boolean_literal() {
    // Verify boolean literal is typed
    let ds = typed_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("r"), ex("boolVal"), var("v"))]),
            expr_eq(
                expr_fn("datatype", vec![expr_var("v")]),
                expr_iri("http://www.w3.org/2001/XMLSchema#boolean"),
            ),
        ),
        vec![variable("r"), variable("v")],
    );
    let test = ConformanceTest::new(
        "type-03",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("type-03 failed");
}

#[test]
fn test_type_04_numeric_comparison() {
    // FILTER(?v > 0) on integer dataset
    let ds = numeric_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            expr_gt(expr_var("v"), expr_lit(lit_int(0))),
        ),
        vec![variable("s"), variable("v")],
    );
    let test = ConformanceTest::new(
        "type-04",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(5), // all values are positive
    );
    runner().run_test(&test).expect("type-04 failed");
}

#[test]
fn test_type_05_negative_number() {
    // FILTER(?v < 0) on dataset with negative number
    let mut ds = InMemoryDataset::new();
    ds.add_triple(
        ex("r1"),
        ex("val"),
        crate::algebra::Term::Literal(crate::algebra::Literal {
            value: "-5".to_string(),
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        }),
    );
    ds.add_triple(ex("r2"), ex("val"), int_lit(10));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            expr_lt(expr_var("v"), expr_lit(lit_int(0))),
        ),
        vec![variable("s"), variable("v")],
    );
    let test = ConformanceTest::new(
        "type-05",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only r1 has -5
    );
    runner().run_test(&test).expect("type-05 failed");
}

#[test]
fn test_type_06_lang_tagged() {
    // FILTER(lang(?label) = "fr")
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("label"), lang_lit("Hello", "en"));
    ds.add_triple(ex("b"), ex("label"), lang_lit("Bonjour", "fr"));

    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("label"), var("label"))]),
            expr_eq(
                expr_fn("lang", vec![expr_var("label")]),
                expr_lit(lit_str("fr")),
            ),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "type-06",
        ConformanceGroup::TypeSystem,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only b
    );
    runner().run_test(&test).expect("type-06 failed");
}
