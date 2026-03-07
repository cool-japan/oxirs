//! VALUES Clause Conformance Tests
//!
//! Tests SPARQL 1.1 VALUES inline data and BIND expressions.

use super::framework::*;
use super::helpers::*;
use crate::algebra::Binding;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// Helper to build a VALUES binding
fn values_binding(pairs: &[(&str, crate::algebra::Term)]) -> Binding {
    let mut binding = Binding::new();
    for (var_name, term) in pairs {
        binding.insert(variable(var_name), term.clone());
    }
    binding
}

// ===== VALUES TESTS =====

#[test]
fn test_values_01_single_var() {
    // VALUES ?s { :a :b :c }
    let ds = values_dataset();

    let vals_algebra = values(
        vec![variable("s")],
        vec![
            values_binding(&[("s", ex("a"))]),
            values_binding(&[("s", ex("b"))]),
            values_binding(&[("s", ex("c"))]),
        ],
    );

    // Join with BGP to get names
    let algebra = project(
        join(
            vals_algebra,
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        ),
        vec![variable("s"), variable("name")],
    );

    let test = ConformanceTest::new(
        "values-01",
        ConformanceGroup::Values,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // a, b, c all have names
    );
    runner().run_test(&test).expect("values-01 failed");
}

#[test]
fn test_values_02_two_vars() {
    // VALUES (?s ?v) { (:a 1) (:b 2) }
    let ds = values_dataset();

    let vals_algebra = values(
        vec![variable("s"), variable("v")],
        vec![
            values_binding(&[("s", ex("a")), ("v", int_lit(1))]),
            values_binding(&[("s", ex("b")), ("v", int_lit(2))]),
        ],
    );

    let test = ConformanceTest::new(
        "values-02",
        ConformanceGroup::Values,
        vals_algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("values-02 failed");
}

#[test]
fn test_values_03_empty_values() {
    // VALUES ?s { } — empty values = no results
    let ds = values_dataset();

    let vals_algebra = values(vec![variable("s")], vec![]);

    let algebra = project(
        join(
            vals_algebra,
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        ),
        vec![variable("name")],
    );

    let test = ConformanceTest::new(
        "values-03",
        ConformanceGroup::Values,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("values-03 failed");
}

#[test]
fn test_values_04_literal_values() {
    // VALUES ?name { "Resource A" "Resource B" }
    let ds = values_dataset();

    let vals_algebra = values(
        vec![variable("name")],
        vec![
            values_binding(&[("name", str_lit("Resource A"))]),
            values_binding(&[("name", str_lit("Resource B"))]),
        ],
    );

    // Join with BGP
    let algebra = project(
        join(
            vals_algebra,
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        ),
        vec![variable("s"), variable("name")],
    );

    let test = ConformanceTest::new(
        "values-04",
        ConformanceGroup::Values,
        algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("values-04 failed");
}

#[test]
fn test_values_05_standalone() {
    // VALUES alone as the complete query: VALUES ?x { 1 2 3 }
    let ds = InMemoryDataset::new();

    let algebra = values(
        vec![variable("x")],
        vec![
            values_binding(&[("x", int_lit(1))]),
            values_binding(&[("x", int_lit(2))]),
            values_binding(&[("x", int_lit(3))]),
        ],
    );

    let test = ConformanceTest::new(
        "values-05",
        ConformanceGroup::Values,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("values-05 failed");
}

#[test]
fn test_values_06_filter_with_values() {
    // VALUES ?id { :a :b :c } . FILTER(?id != :b)
    let ds = values_dataset();

    let vals_algebra = values(
        vec![variable("id")],
        vec![
            values_binding(&[("id", ex("a"))]),
            values_binding(&[("id", ex("b"))]),
            values_binding(&[("id", ex("c"))]),
        ],
    );

    let algebra = filter(
        vals_algebra,
        Expression::Binary {
            op: crate::algebra::BinaryOperator::NotEqual,
            left: Box::new(expr_var("id")),
            right: Box::new(expr_iri("http://example.org/b")),
        },
    );

    let test = ConformanceTest::new(
        "values-06",
        ConformanceGroup::Values,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // a, c
    );
    runner().run_test(&test).expect("values-06 failed");
}

#[test]
fn test_values_07_many_values() {
    // VALUES ?x { 1 2 3 4 5 6 7 8 9 10 }
    let ds = InMemoryDataset::new();

    let bindings: Vec<Binding> = (1..=10)
        .map(|n| values_binding(&[("x", int_lit(n))]))
        .collect();

    let algebra = values(vec![variable("x")], bindings);

    let test = ConformanceTest::new(
        "values-07",
        ConformanceGroup::Values,
        algebra,
        ds,
        ConformanceResult::ResultCount(10),
    );
    runner().run_test(&test).expect("values-07 failed");
}

#[test]
fn test_values_08_join_with_type_filter() {
    // VALUES ?person { :alice :bob } . ?person rdf:type foaf:Person
    let ds = person_dataset();

    let vals_algebra = values(
        vec![variable("person")],
        vec![
            values_binding(&[("person", ex("alice"))]),
            values_binding(&[("person", ex("bob"))]),
        ],
    );

    let algebra = project(
        join(
            vals_algebra,
            bgp(vec![triple(
                var("person"),
                rdf_type(),
                iri(&format!("{FOAF}Person")),
            )]),
        ),
        vec![variable("person")],
    );

    let test = ConformanceTest::new(
        "values-08",
        ConformanceGroup::Values,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // both alice and bob are persons
    );
    runner().run_test(&test).expect("values-08 failed");
}

// ===== BIND TESTS =====

#[test]
fn test_bind_01_basic() {
    // SELECT ?s ?doubled WHERE { ?s :value ?v . BIND(?v * 2 AS ?doubled) }
    // Note: arithmetic in BIND is an Extend with Arithmetic expression
    let ds = numeric_dataset();

    // BIND uses Extend algebra node
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            variable("doubled"),
            Expression::Binary {
                op: crate::algebra::BinaryOperator::Multiply,
                left: Box::new(expr_var("v")),
                right: Box::new(expr_lit(lit_int(2))),
            },
        ),
        vec![variable("s"), variable("doubled")],
    );

    let test = ConformanceTest::new(
        "bind-01",
        ConformanceGroup::Bind,
        algebra,
        ds,
        ConformanceResult::ResultCount(5), // all 5 numeric values
    );
    runner().run_test(&test).expect("bind-01 failed");
}

#[test]
fn test_bind_02_string_constant() {
    // SELECT ?s ?label WHERE { ?s :value ?v . BIND("const" AS ?label) }
    let ds = numeric_dataset();

    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            variable("label"),
            expr_lit(lit_str("constant")),
        ),
        vec![variable("s"), variable("label")],
    );

    let test = ConformanceTest::new(
        "bind-02",
        ConformanceGroup::Bind,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("bind-02 failed");
}

#[test]
fn test_bind_03_var_copy() {
    // SELECT ?s ?copy WHERE { ?s foaf:name ?name . BIND(?name AS ?copy) }
    let ds = person_dataset();

    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            variable("copy"),
            expr_var("name"),
        ),
        vec![variable("s"), variable("copy")],
    );

    let test = ConformanceTest::new(
        "bind-03",
        ConformanceGroup::Bind,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("bind-03 failed");
}

#[test]
fn test_bind_04_iri_bind() {
    // BIND(:category AS ?type)
    let ds = person_dataset();

    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            variable("cat"),
            expr_iri("http://example.org/Person"),
        ),
        vec![variable("s"), variable("cat")],
    );

    let test = ConformanceTest::new(
        "bind-04",
        ConformanceGroup::Bind,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("bind-04 failed");
}

#[test]
fn test_bind_05_chain() {
    // SELECT ?s ?v ?w WHERE { ?s :value ?v . BIND(?v AS ?w) }
    let ds = numeric_dataset();

    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            variable("w"),
            expr_var("v"),
        ),
        vec![variable("s"), variable("v"), variable("w")],
    );

    let test = ConformanceTest::new(
        "bind-05",
        ConformanceGroup::Bind,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("bind-05 failed");
}

#[test]
fn test_values_09_multiple_iri_bindings() {
    // VALUES ?s { :a :b :c :d :e } matching dataset
    let ds = values_dataset();

    let bindings: Vec<Binding> = ["a", "b", "c", "d", "e"]
        .iter()
        .map(|r| values_binding(&[("s", ex(r))]))
        .collect();

    let algebra = project(
        join(
            values(vec![variable("s")], bindings),
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
        ),
        vec![variable("s"), variable("v")],
    );

    let test = ConformanceTest::new(
        "values-09",
        ConformanceGroup::Values,
        algebra,
        ds,
        ConformanceResult::ResultCount(5), // all 5 resources have values
    );
    runner().run_test(&test).expect("values-09 failed");
}
