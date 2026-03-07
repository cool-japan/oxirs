//! Basic Graph Pattern (BGP) Conformance Tests
//!
//! Tests W3C SPARQL 1.1 basic triple pattern matching, join, projection,
//! distinct, ordering, and slice operations.

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== BASIC TRIPLE PATTERN TESTS =====

#[test]
fn test_basic_bgp_1_simple_triple() {
    // SELECT ?name WHERE { ?s foaf:name ?name }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "basic-bgp-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("basic-bgp-1 failed");
}

#[test]
fn test_basic_bgp_2_join_two_patterns() {
    // SELECT ?s ?name ?age WHERE { ?s foaf:name ?name . ?s foaf:age ?age }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![
            triple(var("s"), foaf("name"), var("name")),
            triple(var("s"), iri(&format!("{FOAF}age")), var("age")),
        ]),
        vec![variable("s"), variable("name"), variable("age")],
    );
    let test = ConformanceTest::new(
        "basic-bgp-2",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("basic-bgp-2 failed");
}

#[test]
fn test_basic_bgp_3_type_pattern() {
    // SELECT ?person WHERE { ?person rdf:type foaf:Person }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(
            var("person"),
            rdf_type(),
            iri(&format!("{FOAF}Person")),
        )]),
        vec![variable("person")],
    );
    let test = ConformanceTest::new(
        "basic-bgp-3",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("basic-bgp-3 failed");
}

#[test]
fn test_basic_bgp_4_specific_subject() {
    // SELECT ?p ?o WHERE { :alice ?p ?o }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(ex("alice"), var("p"), var("o"))]),
        vec![variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "basic-bgp-4",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // name, age, type
    );
    runner().run_test(&test).expect("basic-bgp-4 failed");
}

#[test]
fn test_basic_bgp_5_specific_predicate() {
    // SELECT ?s ?o WHERE { ?s foaf:name ?o }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), foaf("name"), var("o"))]),
        vec![variable("s"), variable("o")],
    );
    let test = ConformanceTest::new(
        "basic-bgp-5",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("basic-bgp-5 failed");
}

#[test]
fn test_basic_bgp_6_specific_object() {
    // SELECT ?s WHERE { ?s foaf:name "Alice" }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), foaf("name"), str_lit("Alice"))]),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "basic-bgp-6",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("s", "http://example.org/alice")])]),
    );
    runner().run_test(&test).expect("basic-bgp-6 failed");
}

#[test]
fn test_basic_bgp_7_empty_dataset() {
    // SELECT ?s ?p ?o WHERE { ?s ?p ?o } on empty dataset
    let ds = InMemoryDataset::new();
    let algebra = project(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        vec![variable("s"), variable("p"), variable("o")],
    );
    let test = ConformanceTest::new(
        "basic-bgp-7",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("basic-bgp-7 failed");
}

#[test]
fn test_basic_bgp_8_no_match() {
    // SELECT ?s WHERE { ?s foaf:name "NonExistent" }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), foaf("name"), str_lit("NonExistent"))]),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "basic-bgp-8",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("basic-bgp-8 failed");
}

// ===== DISTINCT TESTS =====

#[test]
fn test_basic_distinct_1() {
    // SELECT DISTINCT ?p WHERE { ?s ?p ?o }
    let ds = person_dataset();
    let algebra = distinct(project(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        vec![variable("p")],
    ));
    let test = ConformanceTest::new(
        "basic-distinct-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // foaf:name, foaf:age, rdf:type
    );
    runner().run_test(&test).expect("basic-distinct-1 failed");
}

#[test]
fn test_basic_distinct_2_names() {
    // SELECT DISTINCT ?name WHERE { ?s foaf:name ?name }
    let ds = person_dataset();
    let algebra = distinct(project(
        bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        vec![variable("name")],
    ));
    let test = ConformanceTest::new(
        "basic-distinct-2",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // 3 distinct names
    );
    runner().run_test(&test).expect("basic-distinct-2 failed");
}

// ===== LIMIT / OFFSET TESTS =====

#[test]
fn test_basic_limit_1() {
    // SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 2
    let ds = person_dataset();
    let algebra = slice(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        None,
        Some(2),
    );
    let test = ConformanceTest::new(
        "basic-limit-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("basic-limit-1 failed");
}

#[test]
fn test_basic_limit_2_zero() {
    // SELECT ?s WHERE { ?s ?p ?o } LIMIT 0
    let ds = person_dataset();
    let algebra = slice(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        None,
        Some(0),
    );
    let test = ConformanceTest::new(
        "basic-limit-2",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("basic-limit-2 failed");
}

#[test]
fn test_basic_offset_1() {
    // SELECT ?s WHERE { ?s ?p ?o } OFFSET 5
    let ds = person_dataset();
    // 9 total triples, offset 5 = 4 results
    let algebra = slice(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        Some(5),
        None,
    );
    let test = ConformanceTest::new(
        "basic-offset-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("basic-offset-1 failed");
}

#[test]
fn test_basic_limit_offset_1() {
    // SELECT ?s WHERE { ?s ?p ?o } LIMIT 3 OFFSET 2
    let ds = person_dataset();
    let algebra = slice(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        Some(2),
        Some(3),
    );
    let test = ConformanceTest::new(
        "basic-limit-offset-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner()
        .run_test(&test)
        .expect("basic-limit-offset-1 failed");
}

// ===== ORDER BY TESTS =====

#[test]
fn test_basic_order_by_1_ascending() {
    // SELECT ?name WHERE { ?s foaf:name ?name } ORDER BY ?name
    let ds = person_dataset();
    let algebra = order_by(
        project(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            vec![variable("name")],
        ),
        vec![asc_cond(expr_var("name"))],
    );
    let test = ConformanceTest::new(
        "basic-order-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::OrderedSelectResults(vec![
            row(&[("name", "Alice")]),
            row(&[("name", "Bob")]),
            row(&[("name", "Charlie")]),
        ]),
    );
    runner().run_test(&test).expect("basic-order-1 failed");
}

#[test]
fn test_basic_order_by_2_descending() {
    // SELECT ?name WHERE { ?s foaf:name ?name } ORDER BY DESC(?name)
    let ds = person_dataset();
    let algebra = order_by(
        project(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            vec![variable("name")],
        ),
        vec![desc_cond(expr_var("name"))],
    );
    let test = ConformanceTest::new(
        "basic-order-2",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::OrderedSelectResults(vec![
            row(&[("name", "Charlie")]),
            row(&[("name", "Bob")]),
            row(&[("name", "Alice")]),
        ]),
    );
    runner().run_test(&test).expect("basic-order-2 failed");
}

#[test]
fn test_basic_order_limit() {
    // SELECT ?name WHERE { ?s foaf:name ?name } ORDER BY ?name LIMIT 2
    let ds = person_dataset();
    let algebra = slice(
        order_by(
            project(
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
                vec![variable("name")],
            ),
            vec![asc_cond(expr_var("name"))],
        ),
        None,
        Some(2),
    );
    let test = ConformanceTest::new(
        "basic-order-limit-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::OrderedSelectResults(vec![
            row(&[("name", "Alice")]),
            row(&[("name", "Bob")]),
        ]),
    );
    runner()
        .run_test(&test)
        .expect("basic-order-limit-1 failed");
}

// ===== FILTER TESTS =====

#[test]
fn test_basic_filter_equality() {
    // SELECT ?s WHERE { ?s foaf:name ?name . FILTER(?name = "Alice") }
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            expr_eq(expr_var("name"), expr_lit(lit_str("Alice"))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "basic-filter-eq-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("s", "http://example.org/alice")])]),
    );
    runner().run_test(&test).expect("basic-filter-eq-1 failed");
}

#[test]
fn test_basic_filter_numeric_gt() {
    // SELECT ?s WHERE { ?s foaf:age ?age . FILTER(?age > 28) }
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(
                var("s"),
                iri(&format!("{FOAF}age")),
                var("age"),
            )]),
            expr_gt(expr_var("age"), expr_lit(lit_int(28))),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "basic-filter-gt-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // alice (30), charlie (35)
    );
    runner().run_test(&test).expect("basic-filter-gt-1 failed");
}

#[test]
fn test_basic_filter_numeric_lt() {
    // SELECT ?s WHERE { ?s foaf:age ?age . FILTER(?age < 30) }
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
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "basic-filter-lt-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // bob (25)
    );
    runner().run_test(&test).expect("basic-filter-lt-1 failed");
}

// ===== STAR PATTERN TESTS =====

#[test]
fn test_basic_all_triples() {
    // SELECT ?s ?p ?o WHERE { ?s ?p ?o }
    let ds = person_dataset();
    let algebra = bgp(vec![triple(var("s"), var("p"), var("o"))]);
    let test = ConformanceTest::new(
        "basic-all-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(9), // 3 people x 3 triples each
    );
    runner().run_test(&test).expect("basic-all-1 failed");
}

#[test]
fn test_basic_three_pattern_join() {
    // SELECT ?s ?name ?age ?type WHERE { ?s foaf:name ?name . ?s foaf:age ?age . ?s rdf:type ?type }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![
            triple(var("s"), foaf("name"), var("name")),
            triple(var("s"), iri(&format!("{FOAF}age")), var("age")),
            triple(var("s"), rdf_type(), var("type")),
        ]),
        vec![
            variable("s"),
            variable("name"),
            variable("age"),
            variable("type"),
        ],
    );
    let test = ConformanceTest::new(
        "basic-three-pattern-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // one result per person
    );
    runner()
        .run_test(&test)
        .expect("basic-three-pattern-1 failed");
}

#[test]
fn test_basic_numeric_dataset_full_scan() {
    // SELECT ?s ?v WHERE { ?s :value ?v }
    let ds = numeric_dataset();
    let algebra = project(
        bgp(vec![triple(var("s"), ex("value"), var("v"))]),
        vec![variable("s"), variable("v")],
    );
    let test = ConformanceTest::new(
        "basic-numeric-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("basic-numeric-1 failed");
}

#[test]
fn test_basic_specific_object_iri() {
    // SELECT ?s WHERE { ?s rdf:type foaf:Person }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(
            var("s"),
            rdf_type(),
            iri(&format!("{FOAF}Person")),
        )]),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "basic-type-iri-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("basic-type-iri-1 failed");
}

#[test]
fn test_basic_project_subset() {
    // SELECT ?name WHERE { ?s foaf:name ?name . ?s foaf:age ?age }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![
            triple(var("s"), foaf("name"), var("name")),
            triple(var("s"), iri(&format!("{FOAF}age")), var("age")),
        ]),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "basic-project-subset-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner()
        .run_test(&test)
        .expect("basic-project-subset-1 failed");
}

// ===== LARGE OFFSET TESTS =====

#[test]
fn test_basic_large_offset() {
    // SELECT ?s WHERE { ?s ?p ?o } OFFSET 100
    let ds = person_dataset();
    let algebra = slice(
        bgp(vec![triple(var("s"), var("p"), var("o"))]),
        Some(100),
        None,
    );
    let test = ConformanceTest::new(
        "basic-large-offset-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(0), // offset beyond all data
    );
    runner()
        .run_test(&test)
        .expect("basic-large-offset-1 failed");
}

#[test]
fn test_basic_named_subject_predicate() {
    // SELECT ?o WHERE { :alice foaf:name ?o }
    let ds = person_dataset();
    let algebra = project(
        bgp(vec![triple(ex("alice"), foaf("name"), var("o"))]),
        vec![variable("o")],
    );
    let test = ConformanceTest::new(
        "basic-named-subj-pred-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("o", "Alice")])]),
    );
    runner()
        .run_test(&test)
        .expect("basic-named-subj-pred-1 failed");
}

#[test]
fn test_basic_blank_dataset() {
    // Simple test on minimal dataset
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("s"), ex("p"), ex("o"));
    let algebra = bgp(vec![triple(var("s"), var("p"), var("o"))]);
    let test = ConformanceTest::new(
        "basic-single-triple-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner()
        .run_test(&test)
        .expect("basic-single-triple-1 failed");
}

#[test]
fn test_basic_filter_and_condition() {
    // SELECT ?s ?age WHERE { ?s foaf:age ?age . FILTER(?age > 20 && ?age < 33) }
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
                expr_lt(expr_var("age"), expr_lit(lit_int(33))),
            ),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "basic-filter-and-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // alice(30), bob(25)
    );
    runner().run_test(&test).expect("basic-filter-and-1 failed");
}

#[test]
fn test_basic_filter_or_condition() {
    // SELECT ?s ?age WHERE { ?s foaf:age ?age . FILTER(?age < 26 || ?age > 33) }
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
                expr_gt(expr_var("age"), expr_lit(lit_int(33))),
            ),
        ),
        vec![variable("s"), variable("age")],
    );
    let test = ConformanceTest::new(
        "basic-filter-or-1",
        ConformanceGroup::BasicPatterns,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // bob(25), charlie(35)
    );
    runner().run_test(&test).expect("basic-filter-or-1 failed");
}
