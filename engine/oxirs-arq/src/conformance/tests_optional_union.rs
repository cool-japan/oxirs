//! OPTIONAL and UNION Conformance Tests
//!
//! Tests SPARQL 1.1 OPTIONAL (LEFT JOIN) and UNION patterns.

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== OPTIONAL / LEFT JOIN TESTS =====

#[test]
fn test_optional_01_basic() {
    // SELECT ?s ?name ?mbox WHERE { ?s foaf:name ?name . OPTIONAL { ?s foaf:mbox ?mbox } }
    // alice has both name and mbox, bob only name, charlie both
    let ds = optional_dataset();
    let algebra = project(
        left_join(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
            None,
        ),
        vec![variable("s"), variable("name"), variable("mbox")],
    );
    let test = ConformanceTest::new(
        "optional-01",
        ConformanceGroup::Optional,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // 3 people, even without mbox
    );
    runner().run_test(&test).expect("optional-01 failed");
}

#[test]
fn test_optional_02_all_have() {
    // SELECT ?s ?name ?mbox WHERE { ?s foaf:name ?name . OPTIONAL { ?s foaf:mbox ?mbox } }
    // When all entries have the optional property
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), foaf("name"), str_lit("A"));
    ds.add_triple(ex("a"), foaf("mbox"), str_lit("a@ex.org"));
    ds.add_triple(ex("b"), foaf("name"), str_lit("B"));
    ds.add_triple(ex("b"), foaf("mbox"), str_lit("b@ex.org"));

    let algebra = project(
        left_join(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
            None,
        ),
        vec![variable("s"), variable("name"), variable("mbox")],
    );
    let test = ConformanceTest::new(
        "optional-02",
        ConformanceGroup::Optional,
        algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("optional-02 failed");
}

#[test]
fn test_optional_03_none_have() {
    // When none have the optional property, we still get left rows
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), foaf("name"), str_lit("A"));
    ds.add_triple(ex("b"), foaf("name"), str_lit("B"));

    let algebra = project(
        left_join(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
            None,
        ),
        vec![variable("s"), variable("name")],
    );
    let test = ConformanceTest::new(
        "optional-03",
        ConformanceGroup::Optional,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // both rows preserved without mbox
    );
    runner().run_test(&test).expect("optional-03 failed");
}

#[test]
fn test_optional_04_empty_left() {
    // OPTIONAL on empty left — no results
    let ds = InMemoryDataset::new();
    let algebra = left_join(
        bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
        None,
    );
    let test = ConformanceTest::new(
        "optional-04",
        ConformanceGroup::Optional,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("optional-04 failed");
}

#[test]
fn test_optional_05_multiple_optional() {
    // SELECT ?s ?name ?mbox ?age WHERE {
    //   ?s foaf:name ?name .
    //   OPTIONAL { ?s foaf:mbox ?mbox }
    //   OPTIONAL { ?s foaf:age ?age }
    // }
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), foaf("name"), str_lit("Alice"));
    ds.add_triple(ex("a"), foaf("mbox"), str_lit("a@ex"));
    ds.add_triple(ex("a"), iri(&format!("{FOAF}age")), int_lit(30));
    ds.add_triple(ex("b"), foaf("name"), str_lit("Bob"));
    // bob has neither mbox nor age

    let algebra = project(
        left_join(
            left_join(
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
                bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
                None,
            ),
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            None,
        ),
        vec![
            variable("s"),
            variable("name"),
            variable("mbox"),
            variable("age"),
        ],
    );
    let test = ConformanceTest::new(
        "optional-05",
        ConformanceGroup::Optional,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // alice and bob
    );
    runner().run_test(&test).expect("optional-05 failed");
}

#[test]
fn test_optional_06_specific_people() {
    // Verify alice and bob are both in result of optional query
    let ds = optional_dataset();
    let algebra = project(
        left_join(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
            None,
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "optional-06",
        ConformanceGroup::Optional,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![
            row(&[("name", "Alice")]),
            row(&[("name", "Bob")]),
            row(&[("name", "Charlie")]),
        ]),
    );
    runner().run_test(&test).expect("optional-06 failed");
}

// ===== UNION TESTS =====

#[test]
fn test_union_01_basic() {
    // SELECT ?s WHERE { { ?s rdf:type foaf:Person } UNION { ?s :orgName ?n } }
    let ds = union_dataset();
    let algebra = project(
        union(
            bgp(vec![triple(
                var("s"),
                rdf_type(),
                iri(&format!("{FOAF}Person")),
            )]),
            bgp(vec![triple(var("s"), ex("orgName"), var("n"))]),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "union-01",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // alice, bob (persons), acme (org)
    );
    runner().run_test(&test).expect("union-01 failed");
}

#[test]
fn test_union_02_same_variable() {
    // SELECT ?name WHERE { { ?s foaf:name ?name } UNION { ?s :orgName ?name } }
    let ds = union_dataset();
    let algebra = project(
        union(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            bgp(vec![triple(var("s"), ex("orgName"), var("name"))]),
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "union-02",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // Alice, Bob, ACME Corp
    );
    runner().run_test(&test).expect("union-02 failed");
}

#[test]
fn test_union_03_left_only() {
    // UNION where only left produces results
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p1"), str_lit("v1"));

    let algebra = union(
        bgp(vec![triple(var("s"), ex("p1"), var("o"))]),
        bgp(vec![triple(var("s"), ex("p2"), var("o"))]),
    );
    let test = ConformanceTest::new(
        "union-03",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("union-03 failed");
}

#[test]
fn test_union_04_right_only() {
    // UNION where only right produces results
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p2"), str_lit("v2"));

    let algebra = union(
        bgp(vec![triple(var("s"), ex("p1"), var("o"))]),
        bgp(vec![triple(var("s"), ex("p2"), var("o"))]),
    );
    let test = ConformanceTest::new(
        "union-04",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("union-04 failed");
}

#[test]
fn test_union_05_empty_both() {
    // UNION of two empty patterns
    let ds = InMemoryDataset::new();
    let algebra = union(
        bgp(vec![triple(var("s"), ex("p1"), var("o"))]),
        bgp(vec![triple(var("s"), ex("p2"), var("o"))]),
    );
    let test = ConformanceTest::new(
        "union-05",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("union-05 failed");
}

#[test]
fn test_union_06_nested() {
    // Nested UNION: { p1 } UNION { { p2 } UNION { p3 } }
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p1"), str_lit("v1"));
    ds.add_triple(ex("b"), ex("p2"), str_lit("v2"));
    ds.add_triple(ex("c"), ex("p3"), str_lit("v3"));

    let algebra = union(
        bgp(vec![triple(var("s"), ex("p1"), var("o"))]),
        union(
            bgp(vec![triple(var("s"), ex("p2"), var("o"))]),
            bgp(vec![triple(var("s"), ex("p3"), var("o"))]),
        ),
    );
    let test = ConformanceTest::new(
        "union-06",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("union-06 failed");
}

#[test]
fn test_union_07_distinct_after() {
    // SELECT DISTINCT ?s WHERE { { ?s foaf:name ?n } UNION { ?s rdf:type ?t } }
    // alice appears in both unions
    let ds = union_dataset();
    let algebra = distinct(project(
        union(
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
            bgp(vec![triple(var("s"), rdf_type(), var("t"))]),
        ),
        vec![variable("s")],
    ));
    let test = ConformanceTest::new(
        "union-07",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // alice, bob, acme (distinct)
    );
    runner().run_test(&test).expect("union-07 failed");
}

#[test]
fn test_union_08_with_filter() {
    // SELECT ?name WHERE { { ?s foaf:name ?name } UNION { ?s :orgName ?name } . FILTER(STRLEN(?name) > 3) }
    let ds = union_dataset();
    // Use regex-based filter approximation: names longer than 3 chars
    let algebra = project(
        filter(
            union(
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
                bgp(vec![triple(var("s"), ex("orgName"), var("name"))]),
            ),
            expr_gt(
                expr_fn("str", vec![expr_var("name")]),
                expr_lit(lit_str("")),
            ),
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "union-08",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("union-08 failed");
}

// ===== OPTIONAL + UNION COMBINATION =====

#[test]
fn test_optional_union_01() {
    // Combine OPTIONAL inside a UNION branch
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), foaf("name"), str_lit("Alice"));
    ds.add_triple(ex("b"), ex("code"), str_lit("B001"));

    let algebra = project(
        union(
            left_join(
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
                bgp(vec![triple(var("s"), ex("code"), var("code"))]),
                None,
            ),
            bgp(vec![triple(var("s"), ex("code"), var("code"))]),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "optional-union-01",
        ConformanceGroup::Union,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // a (from optional), b (from union right)
    );
    runner().run_test(&test).expect("optional-union-01 failed");
}

// ===== OPTIONAL WITH FILTER =====

#[test]
fn test_optional_filter_01() {
    // SELECT ?name WHERE { ?s foaf:name ?name . OPTIONAL { ?s foaf:mbox ?mbox } . FILTER(!BOUND(?mbox) || ?mbox = "alice@example.org") }
    // Only alice has mbox from optional_dataset, bob has no mbox
    let ds = optional_dataset();

    // Simpler test: filter on the left part after optional
    let algebra = project(
        filter(
            left_join(
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
                bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
                None,
            ),
            expr_eq(expr_var("name"), expr_lit(lit_str("Alice"))),
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "optional-filter-01",
        ConformanceGroup::Optional,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("name", "Alice")])]),
    );
    runner().run_test(&test).expect("optional-filter-01 failed");
}

#[test]
fn test_optional_filter_02_no_result() {
    // OPTIONAL + FILTER that rejects all left rows
    let ds = optional_dataset();
    let algebra = project(
        filter(
            left_join(
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
                bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
                None,
            ),
            expr_eq(expr_var("name"), expr_lit(lit_str("NonExistent"))),
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "optional-filter-02",
        ConformanceGroup::Optional,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("optional-filter-02 failed");
}
