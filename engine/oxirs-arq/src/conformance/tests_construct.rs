//! CONSTRUCT Query Conformance Tests
//!
//! Tests SPARQL 1.1 CONSTRUCT queries including:
//! - Template with variables
//! - CONSTRUCT WHERE shorthand
//! - Blank node handling in CONSTRUCT
//! - Graph transformation patterns

use super::framework::*;
use super::helpers::*;
use crate::executor::InMemoryDataset;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== HELPER: Build a construct-style algebra =====
// CONSTRUCT queries in our algebra are modeled as SELECT queries that
// return the triple templates as solutions. The ConstructGraph result
// verifies triple count.

/// Build a dataset with assorted triples for CONSTRUCT tests
fn construct_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        // People
        (ex("alice"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("alice"), foaf("name"), str_lit("Alice")),
        (ex("alice"), foaf("age"), int_lit(30)),
        (ex("alice"), foaf("mbox"), str_lit("alice@example.org")),
        (ex("bob"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("bob"), foaf("name"), str_lit("Bob")),
        (ex("bob"), foaf("age"), int_lit(25)),
        (ex("charlie"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("charlie"), foaf("name"), str_lit("Charlie")),
        (ex("charlie"), foaf("age"), int_lit(35)),
        // Relationships
        (ex("alice"), foaf("knows"), ex("bob")),
        (ex("alice"), foaf("knows"), ex("charlie")),
        (ex("bob"), foaf("knows"), ex("charlie")),
        // Projects
        (ex("proj1"), ex("name"), str_lit("Project Alpha")),
        (ex("proj1"), ex("lead"), ex("alice")),
        (ex("proj2"), ex("name"), str_lit("Project Beta")),
        (ex("proj2"), ex("lead"), ex("bob")),
    ])
}

/// Dataset for CONSTRUCT WHERE shorthand tests
fn small_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("s1"), ex("p"), str_lit("value1")),
        (ex("s2"), ex("p"), str_lit("value2")),
        (ex("s3"), ex("p"), str_lit("value3")),
    ])
}

// ===== BASIC CONSTRUCT TESTS =====

#[test]
fn test_construct_01_basic_template() {
    // CONSTRUCT { ?s a foaf:Person } WHERE { ?s a foaf:Person }
    // Returns triples matching the pattern
    let ds = construct_dataset();
    let algebra = project(
        bgp(vec![triple(
            var("s"),
            rdf_type(),
            iri(&format!("{FOAF}Person")),
        )]),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "construct-01",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // Alice, Bob, Charlie = 3 triples
        ConformanceResult::ConstructGraph(vec![
            (
                "ex:alice".to_string(),
                "rdf:type".to_string(),
                "foaf:Person".to_string(),
            ),
            (
                "ex:bob".to_string(),
                "rdf:type".to_string(),
                "foaf:Person".to_string(),
            ),
            (
                "ex:charlie".to_string(),
                "rdf:type".to_string(),
                "foaf:Person".to_string(),
            ),
        ]),
    );
    runner().run_test(&test).expect("construct-01 failed");
}

#[test]
fn test_construct_02_template_with_name() {
    // CONSTRUCT { ?s foaf:name ?n } WHERE { ?s foaf:name ?n }
    let ds = construct_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-02",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie have foaf:name; proj1/proj2 use ex:name (different predicate) = 3
        ConformanceResult::ConstructGraph(vec![
            (
                "ex:alice".to_string(),
                "foaf:name".to_string(),
                "Alice".to_string(),
            ),
            (
                "ex:bob".to_string(),
                "foaf:name".to_string(),
                "Bob".to_string(),
            ),
            (
                "ex:charlie".to_string(),
                "foaf:name".to_string(),
                "Charlie".to_string(),
            ),
        ]),
    );
    runner().run_test(&test).expect("construct-02 failed");
}

#[test]
fn test_construct_03_where_shorthand() {
    // CONSTRUCT WHERE { ?s ex:p ?o }
    // WHERE shorthand: template == WHERE pattern
    let ds = small_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), ex("p"), var("o"))]),
        vec![variable("s"), variable("o")],
    );
    let test = ConformanceTest::new(
        "construct-03",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![
            (
                "ex:s1".to_string(),
                "ex:p".to_string(),
                "value1".to_string(),
            ),
            (
                "ex:s2".to_string(),
                "ex:p".to_string(),
                "value2".to_string(),
            ),
            (
                "ex:s3".to_string(),
                "ex:p".to_string(),
                "value3".to_string(),
            ),
        ]),
    );
    runner().run_test(&test).expect("construct-03 failed");
}

#[test]
fn test_construct_04_with_filter() {
    // CONSTRUCT { ?s foaf:name ?n } WHERE { ?s foaf:name ?n ; foaf:age ?age FILTER(?age > 28) }
    let ds = construct_dataset();
    let algebra = project(
        filter(
            bgp(vec![
                triple(var("s"), foaf("name"), var("n")),
                triple(var("s"), iri(&format!("{FOAF}age")), var("age")),
            ]),
            expr_gt(
                expr_var("age"),
                Expression::Literal(Literal {
                    value: "28".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            ),
        ),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-04",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice (30), charlie (35) — not bob (25)
        ConformanceResult::ConstructGraph(vec![
            (
                "ex:alice".to_string(),
                "foaf:name".to_string(),
                "Alice".to_string(),
            ),
            (
                "ex:charlie".to_string(),
                "foaf:name".to_string(),
                "Charlie".to_string(),
            ),
        ]),
    );
    runner().run_test(&test).expect("construct-04 failed");
}

#[test]
fn test_construct_05_empty_result() {
    // CONSTRUCT with WHERE clause matching nothing
    let ds = small_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            expr_eq(expr_var("o"), expr_lit(lit_str("nonexistent"))),
        ),
        vec![variable("s"), variable("o")],
    );
    let test = ConformanceTest::new(
        "construct-05",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![]),
    );
    runner().run_test(&test).expect("construct-05 failed");
}

#[test]
fn test_construct_06_with_union() {
    // CONSTRUCT { ?s foaf:name ?n } WHERE { { ?s foaf:name ?n } UNION { ?s ex:name ?n } }
    let ds = construct_dataset();
    let algebra = project(
        union(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            bgp(vec![triple(var("s"), ex("name"), var("n"))]),
        ),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-06",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie (foaf:name) + proj1, proj2 (ex:name) = 5
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()), // placeholder 1
            ("".to_string(), "".to_string(), "".to_string()), // placeholder 2
            ("".to_string(), "".to_string(), "".to_string()), // placeholder 3
            ("".to_string(), "".to_string(), "".to_string()), // placeholder 4
            ("".to_string(), "".to_string(), "".to_string()), // placeholder 5
        ]),
    );
    runner().run_test(&test).expect("construct-06 failed");
}

#[test]
fn test_construct_07_with_optional() {
    // CONSTRUCT { ?s foaf:name ?n ; foaf:mbox ?mb } WHERE { ?s foaf:name ?n OPTIONAL { ?s foaf:mbox ?mb } }
    let ds = construct_dataset();
    let algebra = project(
        left_join(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            bgp(vec![triple(var("s"), foaf("mbox"), var("mb"))]),
            None,
        ),
        vec![variable("s"), variable("n"), variable("mb")],
    );
    let test = ConformanceTest::new(
        "construct-07",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice (foaf:name + mbox), bob (foaf:name only), charlie (foaf:name only) = 3
        // proj1/proj2 use ex:name, not foaf:name, so they don't match the left BGP
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-07 failed");
}

#[test]
fn test_construct_08_distinct_construct() {
    // CONSTRUCT DISTINCT — deduplicated results
    let ds = construct_dataset();
    let algebra = project(
        distinct(bgp(vec![triple(var("s"), rdf_type(), var("t"))])),
        vec![variable("s"), variable("t")],
    );
    let test = ConformanceTest::new(
        "construct-08",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // Alice, Bob, Charlie rdf:type foaf:Person = 3 distinct
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-08 failed");
}

#[test]
fn test_construct_09_with_limit() {
    // CONSTRUCT { ?s ex:p ?o } WHERE { ?s ex:p ?o } LIMIT 2
    let ds = small_dataset();
    let algebra = project(
        slice(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            None,
            Some(2),
        ),
        vec![variable("s"), variable("o")],
    );
    let test = ConformanceTest::new(
        "construct-09",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-09 failed");
}

#[test]
fn test_construct_10_with_offset() {
    // CONSTRUCT OFFSET 1 — skip first
    let ds = small_dataset();
    let algebra = project(
        slice(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            Some(1),
            None,
        ),
        vec![variable("s"), variable("o")],
    );
    let test = ConformanceTest::new(
        "construct-10",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-10 failed");
}

// ===== CONSTRUCT WITH JOINS =====

#[test]
fn test_construct_11_join_pattern() {
    // CONSTRUCT { ?person foaf:knows ?friend } WHERE { ?person foaf:knows ?friend }
    let ds = construct_dataset();
    let algebra = project(
        bgp(vec![triple(var("person"), foaf("knows"), var("friend"))]),
        vec![variable("person"), variable("friend")],
    );
    let test = ConformanceTest::new(
        "construct-11",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice-bob, alice-charlie, bob-charlie = 3
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-11 failed");
}

#[test]
fn test_construct_12_multi_pattern() {
    // CONSTRUCT { ?s foaf:name ?n ; foaf:age ?a } WHERE { ?s foaf:name ?n ; foaf:age ?a }
    let ds = construct_dataset();
    let algebra = project(
        bgp(vec![
            triple(var("s"), foaf("name"), var("n")),
            triple(var("s"), iri(&format!("{FOAF}age")), var("a")),
        ]),
        vec![variable("s"), variable("n"), variable("a")],
    );
    let test = ConformanceTest::new(
        "construct-12",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie each have name and age = 3
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-12 failed");
}

#[test]
fn test_construct_13_project_lead() {
    // CONSTRUCT { ?proj ex:lead ?lead } WHERE { ?proj ex:lead ?lead }
    let ds = construct_dataset();
    let algebra = project(
        bgp(vec![triple(var("proj"), ex("lead"), var("lead"))]),
        vec![variable("proj"), variable("lead")],
    );
    let test = ConformanceTest::new(
        "construct-13",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // proj1-alice, proj2-bob = 2
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-13 failed");
}

// ===== CONSTRUCT WITH VALUES =====

#[test]
fn test_construct_14_with_values() {
    // CONSTRUCT { ?s foaf:name ?n } WHERE { VALUES ?s { ex:alice ex:bob } ?s foaf:name ?n }
    let ds = construct_dataset();
    let algebra = project(
        join(
            values(
                vec![variable("s")],
                vec![
                    {
                        let mut b = crate::algebra::Binding::new();
                        b.insert(variable("s"), ex("alice"));
                        b
                    },
                    {
                        let mut b = crate::algebra::Binding::new();
                        b.insert(variable("s"), ex("bob"));
                        b
                    },
                ],
            ),
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
        ),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-14",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice-Alice, bob-Bob = 2
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-14 failed");
}

// ===== CONSTRUCT TEMPLATE TRANSFORMATIONS =====

#[test]
fn test_construct_15_bind_transform() {
    // CONSTRUCT { ?s ex:nameLen ?len } WHERE { ?s foaf:name ?n BIND(strlen(?n) AS ?len) }
    let ds = construct_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            variable("len"),
            expr_fn("strlen", vec![expr_var("n")]),
        ),
        vec![variable("s"), variable("n"), variable("len")],
    );
    let test = ConformanceTest::new(
        "construct-15",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie have foaf:name; proj1/proj2 use ex:name = 3
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-15 failed");
}

#[test]
fn test_construct_16_order_by_then_construct() {
    // CONSTRUCT { ?s foaf:name ?n } WHERE { ?s foaf:name ?n } ORDER BY ?n
    let ds = construct_dataset();
    let algebra = project(
        order_by(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            vec![asc_cond(expr_var("n"))],
        ),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-16",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie have foaf:name; proj1/proj2 use ex:name = 3 ordered alphabetically
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-16 failed");
}

// ===== CONSTRUCT BLANK NODES =====

#[test]
fn test_construct_17_blank_node_subjects() {
    // Dataset with blank node subjects
    let mut ds = InMemoryDataset::new();
    let bnode1 = crate::algebra::Term::BlankNode("b1".to_string());
    let bnode2 = crate::algebra::Term::BlankNode("b2".to_string());
    ds.add_triple(bnode1.clone(), foaf("name"), str_lit("Anonymous1"));
    ds.add_triple(bnode2.clone(), foaf("name"), str_lit("Anonymous2"));
    ds.add_triple(bnode1, rdf_type(), iri(&format!("{FOAF}Person")));

    let algebra = project(
        bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-17",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // b1 and b2 each have name = 2
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-17 failed");
}

#[test]
fn test_construct_18_blank_node_objects() {
    // Dataset with blank node objects
    let mut ds = InMemoryDataset::new();
    let bnode = crate::algebra::Term::BlankNode("addr1".to_string());
    ds.add_triple(ex("alice"), ex("address"), bnode.clone());
    ds.add_triple(bnode.clone(), ex("street"), str_lit("123 Main St"));
    ds.add_triple(bnode, ex("city"), str_lit("Springfield"));

    let algebra = project(
        bgp(vec![triple(var("s"), ex("address"), var("addr"))]),
        vec![variable("s"), variable("addr")],
    );
    let test = ConformanceTest::new(
        "construct-18",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice has one address
        ConformanceResult::ConstructGraph(vec![("".to_string(), "".to_string(), "".to_string())]),
    );
    runner().run_test(&test).expect("construct-18 failed");
}

// ===== NESTED CONSTRUCT PATTERNS =====

#[test]
fn test_construct_19_nested_join() {
    // Join two BGPs for CONSTRUCT
    let ds = construct_dataset();
    let algebra = project(
        join(
            bgp(vec![triple(var("proj"), ex("lead"), var("lead"))]),
            bgp(vec![triple(var("lead"), foaf("name"), var("lead_name"))]),
        ),
        vec![variable("proj"), variable("lead"), variable("lead_name")],
    );
    let test = ConformanceTest::new(
        "construct-19",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // proj1→alice→Alice, proj2→bob→Bob = 2
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-19 failed");
}

#[test]
fn test_construct_20_full_pipeline() {
    // Full pipeline: join + filter + order + limit for CONSTRUCT
    let ds = construct_dataset();
    let algebra = project(
        slice(
            order_by(
                filter(
                    bgp(vec![
                        triple(var("s"), rdf_type(), iri(&format!("{FOAF}Person"))),
                        triple(var("s"), foaf("name"), var("n")),
                        triple(var("s"), iri(&format!("{FOAF}age")), var("age")),
                    ]),
                    expr_gt(
                        expr_var("age"),
                        Expression::Literal(Literal {
                            value: "24".to_string(),
                            language: None,
                            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )),
                        }),
                    ),
                ),
                vec![asc_cond(expr_var("n"))],
            ),
            None,
            Some(2),
        ),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-20",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice (30) and bob (25) — ordered by name, limit 2
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-20 failed");
}

// ===== ADDITIONAL CONSTRUCT EDGE CASES =====

#[test]
fn test_construct_21_single_triple() {
    // Single triple result
    let ds = small_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            expr_eq(expr_var("s"), expr_iri("http://example.org/s1")),
        ),
        vec![variable("s"), variable("o")],
    );
    let test = ConformanceTest::new(
        "construct-21",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![(
            "ex:s1".to_string(),
            "ex:p".to_string(),
            "value1".to_string(),
        )]),
    );
    runner().run_test(&test).expect("construct-21 failed");
}

#[test]
fn test_construct_22_all_subject_types() {
    // Dataset with IRI, blank node, and literal subjects (unusual but valid)
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("r1"), ex("pred"), int_lit(1));
    ds.add_triple(ex("r2"), ex("pred"), int_lit(2));
    ds.add_triple(
        crate::algebra::Term::BlankNode("bn1".to_string()),
        ex("pred"),
        int_lit(3),
    );

    let algebra = project(
        bgp(vec![triple(var("s"), ex("pred"), var("v"))]),
        vec![variable("s"), variable("v")],
    );
    let test = ConformanceTest::new(
        "construct-22",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-22 failed");
}

#[test]
fn test_construct_23_type_check_in_where() {
    // CONSTRUCT { ?s foaf:name ?n } WHERE { ?s foaf:name ?n FILTER(isIRI(?s)) }
    let ds = construct_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            Expression::Unary {
                op: crate::algebra::UnaryOperator::IsIri,
                operand: Box::new(expr_var("s")),
            },
        ),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-23",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie have foaf:name with IRI subjects = 3
        // (proj1/proj2 use ex:name, not foaf:name)
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-23 failed");
}

#[test]
fn test_construct_24_nested_optional_blank() {
    // CONSTRUCT from nested optional with blank nodes
    let mut ds = InMemoryDataset::new();
    let addr = crate::algebra::Term::BlankNode("addr".to_string());
    ds.add_triple(ex("alice"), foaf("name"), str_lit("Alice"));
    ds.add_triple(ex("alice"), ex("address"), addr.clone());
    ds.add_triple(addr, ex("city"), str_lit("Springfield"));
    ds.add_triple(ex("bob"), foaf("name"), str_lit("Bob"));
    // bob has no address

    let algebra = project(
        left_join(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            bgp(vec![triple(var("s"), ex("address"), var("addr"))]),
            None,
        ),
        vec![variable("s"), variable("n"), variable("addr")],
    );
    let test = ConformanceTest::new(
        "construct-24",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice (with addr), bob (without addr) = 2 solutions
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-24 failed");
}

#[test]
fn test_construct_25_subquery_in_construct() {
    // CONSTRUCT { ?s foaf:name ?n } WHERE { SELECT ?s ?n WHERE { ?s foaf:name ?n } LIMIT 3 }
    let ds = construct_dataset();
    // Model the subquery as a slice inside a project
    let inner = project(
        slice(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            None,
            Some(3),
        ),
        vec![variable("s"), variable("n")],
    );
    let algebra = project(inner, vec![variable("s"), variable("n")]);
    let test = ConformanceTest::new(
        "construct-25",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-25 failed");
}

#[test]
fn test_construct_26_full_graph_copy() {
    // Copy all triples from small_dataset
    let ds = small_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        vec![variable("s"), variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "construct-26",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // 3 triples in small_dataset
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-26 failed");
}

#[test]
fn test_construct_27_with_str_function() {
    // CONSTRUCT uses STR() result
    let ds = small_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            variable("os"),
            expr_fn("str", vec![expr_var("o")]),
        ),
        vec![variable("s"), variable("os")],
    );
    let test = ConformanceTest::new(
        "construct-27",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-27 failed");
}

#[test]
fn test_construct_28_minus_pattern() {
    // CONSTRUCT { ?s foaf:name ?n } WHERE { ?s foaf:name ?n MINUS { ?s foaf:mbox ?mb } }
    let ds = construct_dataset();
    let algebra = project(
        crate::algebra::Algebra::Minus {
            left: Box::new(bgp(vec![triple(var("s"), foaf("name"), var("n"))])),
            right: Box::new(bgp(vec![triple(var("s"), foaf("mbox"), var("mb"))])),
        },
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "construct-28",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // foaf:name matches alice/bob/charlie (3); alice has mbox so is excluded by MINUS.
        // proj1/proj2 use ex:name, not foaf:name, so they are not in the left side. Result = 2.
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-28 failed");
}

#[test]
fn test_construct_29_reduced_results() {
    // CONSTRUCT REDUCED — may remove duplicates
    let ds = construct_dataset();
    let algebra = project(
        crate::algebra::Algebra::Reduced {
            pattern: Box::new(bgp(vec![triple(var("s"), rdf_type(), var("t"))])),
        },
        vec![variable("s"), variable("t")],
    );
    let test = ConformanceTest::new(
        "construct-29",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
            ("".to_string(), "".to_string(), "".to_string()),
        ]),
    );
    runner().run_test(&test).expect("construct-29 failed");
}

#[test]
fn test_construct_30_empty_dataset() {
    // CONSTRUCT from empty dataset yields empty graph
    let ds = InMemoryDataset::new();
    let algebra = project(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        vec![variable("s"), variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "construct-30",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ConstructGraph(vec![]),
    );
    runner().run_test(&test).expect("construct-30 failed");
}
