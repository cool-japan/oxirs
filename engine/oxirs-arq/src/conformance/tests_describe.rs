//! DESCRIBE Query Conformance Tests
//!
//! Tests SPARQL 1.1 DESCRIBE queries including:
//! - DESCRIBE <IRI>
//! - DESCRIBE ?var WHERE {}
//! - Multiple subjects
//! - Description strategies (CBD, symmetric CBD, labeled CBD)

use super::framework::*;
use super::helpers::*;
use crate::executor::InMemoryDataset;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== HELPERS =====

/// Dataset for DESCRIBE tests
fn describe_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        // Alice
        (ex("alice"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("alice"), foaf("name"), str_lit("Alice")),
        (ex("alice"), foaf("age"), int_lit(30)),
        (ex("alice"), foaf("mbox"), str_lit("alice@example.org")),
        (ex("alice"), foaf("knows"), ex("bob")),
        // Bob
        (ex("bob"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("bob"), foaf("name"), str_lit("Bob")),
        (ex("bob"), foaf("age"), int_lit(25)),
        // Charlie
        (ex("charlie"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("charlie"), foaf("name"), str_lit("Charlie")),
        (ex("charlie"), foaf("age"), int_lit(35)),
        // Organisation
        (ex("acme"), rdf_type(), ex("Organisation")),
        (ex("acme"), ex("orgName"), str_lit("ACME Corp")),
        (ex("acme"), ex("founder"), ex("alice")),
        // Products
        (ex("prod1"), ex("name"), str_lit("Widget")),
        (ex("prod1"), ex("price"), int_lit(10)),
        (ex("prod1"), ex("madeBy"), ex("acme")),
        (ex("prod2"), ex("name"), str_lit("Gadget")),
        (ex("prod2"), ex("price"), int_lit(20)),
        (ex("prod2"), ex("madeBy"), ex("acme")),
    ])
}

// ===== DESCRIBE <IRI> TESTS =====

#[test]
fn test_describe_01_iri_all_props() {
    // DESCRIBE ex:alice — returns all triples with alice as subject
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(ex("alice"), var("p"), var("o"))]),
        vec![variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "describe-01",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice has: type, name, age, mbox, knows = 5 triples
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("describe-01 failed");
}

#[test]
fn test_describe_02_iri_bob() {
    // DESCRIBE ex:bob — returns all triples with bob as subject
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(ex("bob"), var("p"), var("o"))]),
        vec![variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "describe-02",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // bob has: type, name, age = 3 triples
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("describe-02 failed");
}

#[test]
fn test_describe_03_iri_acme() {
    // DESCRIBE ex:acme — returns organisation triples
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(ex("acme"), var("p"), var("o"))]),
        vec![variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "describe-03",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // acme has: type, orgName, founder = 3 triples
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("describe-03 failed");
}

// ===== DESCRIBE ?var WHERE {} TESTS =====

#[test]
fn test_describe_04_var_all_persons() {
    // DESCRIBE ?s WHERE { ?s a foaf:Person }
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(
            var("s"),
            rdf_type(),
            iri(&format!("{FOAF}Person")),
        )]),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-04",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie = 3 subjects
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("describe-04 failed");
}

#[test]
fn test_describe_05_var_with_filter() {
    // DESCRIBE ?s WHERE { ?s foaf:age ?age FILTER(?age > 28) }
    let ds = describe_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
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
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-05",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice (30), charlie (35) — not bob (25) = 2
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("describe-05 failed");
}

#[test]
fn test_describe_06_var_where_name() {
    // DESCRIBE ?s WHERE { ?s foaf:name "Alice" }
    let ds = describe_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            expr_eq(expr_var("n"), expr_lit(lit_str("Alice"))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-06",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("describe-06 failed");
}

// ===== MULTIPLE SUBJECTS =====

#[test]
fn test_describe_07_multiple_iris() {
    // DESCRIBE ex:alice ex:bob — retrieve triples for both IRIs combined
    let ds = describe_dataset();
    let algebra = project(
        union(
            bgp(vec![triple(ex("alice"), var("p"), var("o"))]),
            bgp(vec![triple(ex("bob"), var("p"), var("o"))]),
        ),
        vec![variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "describe-07",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice: 5, bob: 3 = 8 triples
        ConformanceResult::ResultCount(8),
    );
    runner().run_test(&test).expect("describe-07 failed");
}

#[test]
fn test_describe_08_multiple_vars() {
    // DESCRIBE ?s ?o WHERE { ?s foaf:knows ?o }
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), foaf("knows"), var("o"))]),
        vec![variable("s"), variable("o")],
    );
    let test = ConformanceTest::new(
        "describe-08",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice knows bob = 1 knows triple
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("describe-08 failed");
}

#[test]
fn test_describe_09_describe_three_subjects() {
    // DESCRIBE ex:alice ex:bob ex:charlie
    let ds = describe_dataset();
    let algebra = project(
        union(
            union(
                bgp(vec![triple(ex("alice"), var("p1"), var("o1"))]),
                bgp(vec![triple(ex("bob"), var("p2"), var("o2"))]),
            ),
            bgp(vec![triple(ex("charlie"), var("p3"), var("o3"))]),
        ),
        vec![variable("p1"), variable("o1")],
    );
    let test = ConformanceTest::new(
        "describe-09",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice: 5, bob: 3, charlie: 3 = 11 total across all union branches
        ConformanceResult::ResultCount(11),
    );
    runner().run_test(&test).expect("describe-09 failed");
}

// ===== DESCRIBE WITH PATTERNS =====

#[test]
fn test_describe_10_with_join() {
    // DESCRIBE ?person WHERE { ?org ex:founder ?person }
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(var("org"), ex("founder"), var("person"))]),
        vec![variable("person")],
    );
    let test = ConformanceTest::new(
        "describe-10",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // acme has alice as founder = 1
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("describe-10 failed");
}

#[test]
fn test_describe_11_with_optional() {
    // DESCRIBE ?s WHERE { ?s foaf:name ?n OPTIONAL { ?s foaf:mbox ?m } }
    let ds = describe_dataset();
    let algebra = project(
        left_join(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            bgp(vec![triple(var("s"), foaf("mbox"), var("m"))]),
            None,
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-11",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie have foaf:name (3); prod1/prod2 use ex:name (different predicate) = 3
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("describe-11 failed");
}

#[test]
fn test_describe_12_with_union_pattern() {
    // DESCRIBE ?s WHERE { { ?s a foaf:Person } UNION { ?s a ex:Organisation } }
    let ds = describe_dataset();
    let algebra = project(
        union(
            bgp(vec![triple(
                var("s"),
                rdf_type(),
                iri(&format!("{FOAF}Person")),
            )]),
            bgp(vec![triple(var("s"), rdf_type(), ex("Organisation"))]),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-12",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie, acme = 4
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("describe-12 failed");
}

// ===== DESCRIBE PROPERTIES (CBD Strategy) =====

#[test]
fn test_describe_13_properties_of_subject() {
    // Retrieve all properties of a known subject
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(ex("prod1"), var("p"), var("o"))]),
        vec![variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "describe-13",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // prod1: name, price, madeBy = 3
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("describe-13 failed");
}

#[test]
fn test_describe_14_properties_of_all_products() {
    // DESCRIBE ?p WHERE { ?p ex:madeBy ex:acme }
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(var("p"), ex("madeBy"), ex("acme"))]),
        vec![variable("p")],
    );
    let test = ConformanceTest::new(
        "describe-14",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // prod1 and prod2 made by acme = 2
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("describe-14 failed");
}

#[test]
fn test_describe_15_describe_limit() {
    // DESCRIBE ?s WHERE { ?s a foaf:Person } LIMIT 2
    let ds = describe_dataset();
    let algebra = project(
        slice(
            bgp(vec![triple(
                var("s"),
                rdf_type(),
                iri(&format!("{FOAF}Person")),
            )]),
            None,
            Some(2),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-15",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("describe-15 failed");
}

// ===== DESCRIBE WITH ORDER BY =====

#[test]
fn test_describe_16_order_by_name() {
    // DESCRIBE ?s WHERE { ?s foaf:name ?n } ORDER BY ?n
    let ds = describe_dataset();
    let algebra = project(
        order_by(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            vec![asc_cond(expr_var("n"))],
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-16",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice, bob, charlie have foaf:name (3); prod1/prod2 use ex:name = 3
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("describe-16 failed");
}

#[test]
fn test_describe_17_desc_order() {
    // DESCRIBE ?s WHERE { ?s foaf:age ?age } ORDER BY DESC(?age)
    let ds = describe_dataset();
    let algebra = project(
        order_by(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            vec![desc_cond(expr_var("age"))],
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-17",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // charlie (35), alice (30), bob (25) = 3
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("describe-17 failed");
}

// ===== DESCRIBE BLANK NODES =====

#[test]
fn test_describe_18_blank_node_subject() {
    // DESCRIBE with blank node in dataset
    let mut ds = InMemoryDataset::new();
    let bnode = crate::algebra::Term::BlankNode("b1".to_string());
    ds.add_triple(bnode.clone(), foaf("name"), str_lit("Anonymous"));
    ds.add_triple(bnode.clone(), rdf_type(), iri(&format!("{FOAF}Person")));

    let algebra = project(
        bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
        vec![variable("s"), variable("n")],
    );
    let test = ConformanceTest::new(
        "describe-18",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("describe-18 failed");
}

#[test]
fn test_describe_19_blank_node_object() {
    // DESCRIBE where object is blank node
    let mut ds = InMemoryDataset::new();
    let bnode = crate::algebra::Term::BlankNode("addr".to_string());
    ds.add_triple(ex("alice"), ex("address"), bnode.clone());
    ds.add_triple(bnode.clone(), ex("city"), str_lit("Springfield"));
    ds.add_triple(bnode, ex("zip"), str_lit("62701"));

    let algebra = project(
        bgp(vec![triple(ex("alice"), ex("address"), var("addr"))]),
        vec![variable("addr")],
    );
    let test = ConformanceTest::new(
        "describe-19",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("describe-19 failed");
}

// ===== DESCRIBE EMPTY PATTERNS =====

#[test]
fn test_describe_20_no_match() {
    // DESCRIBE ?s WHERE { ?s ex:nonExistent ?o } — no match
    let ds = describe_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), ex("nonExistent"), var("o"))]),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-20",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("describe-20 failed");
}

#[test]
fn test_describe_21_empty_dataset() {
    // DESCRIBE against empty dataset
    let ds = InMemoryDataset::new();
    let algebra = project(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-21",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("describe-21 failed");
}

// ===== DESCRIBE WITH ADVANCED PATTERNS =====

#[test]
fn test_describe_22_values_driven() {
    // DESCRIBE ?s WHERE { VALUES ?s { ex:alice ex:charlie } }
    let ds = describe_dataset();
    let alice_binding = {
        let mut b = crate::algebra::Binding::new();
        b.insert(variable("s"), ex("alice"));
        b
    };
    let charlie_binding = {
        let mut b = crate::algebra::Binding::new();
        b.insert(variable("s"), ex("charlie"));
        b
    };
    let algebra = project(
        values(vec![variable("s")], vec![alice_binding, charlie_binding]),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-22",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        // alice and charlie from VALUES = 2
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("describe-22 failed");
}

#[test]
fn test_describe_23_subquery_driven() {
    // DESCRIBE ?s WHERE { SELECT ?s WHERE { ?s a foaf:Person } LIMIT 1 }
    let ds = describe_dataset();
    let algebra = project(
        slice(
            bgp(vec![triple(
                var("s"),
                rdf_type(),
                iri(&format!("{FOAF}Person")),
            )]),
            None,
            Some(1),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "describe-23",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("describe-23 failed");
}
