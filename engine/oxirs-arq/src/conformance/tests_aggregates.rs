//! Aggregate Conformance Tests
//!
//! Tests SPARQL 1.1 aggregate functions: COUNT, SUM, AVG, MIN, MAX,
//! GROUP_CONCAT, GROUP BY, and HAVING clauses.

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== COUNT TESTS =====

#[test]
fn test_agg_01_count_star() {
    // SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }
    let ds = person_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), var("p"), var("o"))]),
            vec![],
            vec![(variable("count"), agg_count_star())],
        ),
        vec![variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-01",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("count", "9")])]),
    );
    runner().run_test(&test).expect("agg-01 failed");
}

#[test]
fn test_agg_02_count_var() {
    // SELECT (COUNT(?name) AS ?count) WHERE { ?s foaf:name ?name }
    let ds = person_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            vec![],
            vec![(variable("count"), agg_count("name"))],
        ),
        vec![variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-02",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("count", "3")])]),
    );
    runner().run_test(&test).expect("agg-02 failed");
}

#[test]
fn test_agg_03_count_empty() {
    // SELECT (COUNT(*) AS ?count) WHERE { ?s foaf:nonexistent ?o }
    let ds = person_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), ex("nonexistent"), var("o"))]),
            vec![],
            vec![(variable("count"), agg_count_star())],
        ),
        vec![variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-03",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("count", "0")])]),
    );
    runner().run_test(&test).expect("agg-03 failed");
}

#[test]
fn test_agg_04_sum() {
    // SELECT (SUM(?v) AS ?total) WHERE { ?s :value ?v }
    let ds = numeric_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            vec![],
            vec![(variable("total"), agg_sum("v"))],
        ),
        vec![variable("total")],
    );
    let test = ConformanceTest::new(
        "agg-04",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("total", "100")])]),
    );
    runner().run_test(&test).expect("agg-04 failed");
}

#[test]
fn test_agg_05_avg() {
    // SELECT (AVG(?v) AS ?avg) WHERE { ?s :value ?v }
    let ds = numeric_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            vec![],
            vec![(variable("avg"), agg_avg("v"))],
        ),
        vec![variable("avg")],
    );
    let test = ConformanceTest::new(
        "agg-05",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("avg", "20")])]),
    );
    runner().run_test(&test).expect("agg-05 failed");
}

#[test]
fn test_agg_06_min() {
    // SELECT (MIN(?v) AS ?min) WHERE { ?s :value ?v }
    let ds = numeric_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            vec![],
            vec![(variable("min"), agg_min("v"))],
        ),
        vec![variable("min")],
    );
    let test = ConformanceTest::new(
        "agg-06",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("min", "10")])]),
    );
    runner().run_test(&test).expect("agg-06 failed");
}

#[test]
fn test_agg_07_max() {
    // SELECT (MAX(?v) AS ?max) WHERE { ?s :value ?v }
    let ds = numeric_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            vec![],
            vec![(variable("max"), agg_max("v"))],
        ),
        vec![variable("max")],
    );
    let test = ConformanceTest::new(
        "agg-07",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("max", "30")])]),
    );
    runner().run_test(&test).expect("agg-07 failed");
}

#[test]
fn test_agg_08_group_concat_default_sep() {
    // SELECT (GROUP_CONCAT(?name) AS ?names) WHERE { ?s foaf:name ?name }
    // Default separator is " "
    let ds = person_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            vec![],
            vec![(variable("names"), agg_group_concat("name", None))],
        ),
        vec![variable("names")],
    );
    let test = ConformanceTest::new(
        "agg-08",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // one row with concatenated names
    );
    runner().run_test(&test).expect("agg-08 failed");
}

#[test]
fn test_agg_09_group_concat_custom_sep() {
    // SELECT (GROUP_CONCAT(?name; SEPARATOR=",") AS ?names) WHERE { ?s foaf:name ?name }
    let ds = person_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            vec![],
            vec![(
                variable("names"),
                agg_group_concat("name", Some(",".to_string())),
            )],
        ),
        vec![variable("names")],
    );
    let test = ConformanceTest::new(
        "agg-09",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("agg-09 failed");
}

// ===== GROUP BY TESTS =====

#[test]
fn test_agg_group_01_by_category() {
    // SELECT ?cat (COUNT(*) AS ?count) WHERE { ?item :category ?cat } GROUP BY ?cat
    let ds = group_by_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("item"), ex("category"), var("cat"))]),
            vec![group_var("cat")],
            vec![(variable("count"), agg_count_star())],
        ),
        vec![variable("cat"), variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-group-01",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // 2 categories: A and B
    );
    runner().run_test(&test).expect("agg-group-01 failed");
}

#[test]
fn test_agg_group_02_sum_by_category() {
    // SELECT ?cat (SUM(?price) AS ?total) WHERE { ?item :category ?cat . ?item :price ?price } GROUP BY ?cat
    let ds = group_by_dataset();
    let algebra = project(
        group(
            bgp(vec![
                triple(var("item"), ex("category"), var("cat")),
                triple(var("item"), ex("price"), var("price")),
            ]),
            vec![group_var("cat")],
            vec![(variable("total"), agg_sum("price"))],
        ),
        vec![variable("cat"), variable("total")],
    );
    let test = ConformanceTest::new(
        "agg-group-02",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // A and B categories
    );
    runner().run_test(&test).expect("agg-group-02 failed");
}

#[test]
fn test_agg_group_03_avg_by_category() {
    // SELECT ?cat (AVG(?price) AS ?avg) WHERE { ?item :category ?cat . ?item :price ?price } GROUP BY ?cat
    let ds = group_by_dataset();
    let algebra = project(
        group(
            bgp(vec![
                triple(var("item"), ex("category"), var("cat")),
                triple(var("item"), ex("price"), var("price")),
            ]),
            vec![group_var("cat")],
            vec![(variable("avg"), agg_avg("price"))],
        ),
        vec![variable("cat"), variable("avg")],
    );
    let test = ConformanceTest::new(
        "agg-group-03",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("agg-group-03 failed");
}

#[test]
fn test_agg_group_04_min_max_by_category() {
    // SELECT ?cat (MIN(?price) AS ?min) (MAX(?price) AS ?max) WHERE { ?item :category ?cat . ?item :price ?price } GROUP BY ?cat
    let ds = group_by_dataset();
    let algebra = project(
        group(
            bgp(vec![
                triple(var("item"), ex("category"), var("cat")),
                triple(var("item"), ex("price"), var("price")),
            ]),
            vec![group_var("cat")],
            vec![
                (variable("min"), agg_min("price")),
                (variable("max"), agg_max("price")),
            ],
        ),
        vec![variable("cat"), variable("min"), variable("max")],
    );
    let test = ConformanceTest::new(
        "agg-group-04",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("agg-group-04 failed");
}

#[test]
fn test_agg_group_05_single_group() {
    // SELECT (COUNT(*) AS ?count) WHERE { ?s :value ?v }
    // No GROUP BY — single group
    let ds = numeric_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            vec![],
            vec![(variable("count"), agg_count_star())],
        ),
        vec![variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-group-05",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // one group
    );
    runner().run_test(&test).expect("agg-group-05 failed");
}

// ===== COUNT DISTINCT TESTS =====

#[test]
fn test_agg_count_distinct_01() {
    // SELECT (COUNT(DISTINCT ?cat) AS ?cats) WHERE { ?item :category ?cat }
    let ds = group_by_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("item"), ex("category"), var("cat"))]),
            vec![],
            vec![(variable("cats"), agg_count_distinct("cat"))],
        ),
        vec![variable("cats")],
    );
    let test = ConformanceTest::new(
        "agg-count-distinct-01",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("cats", "2")])]),
    );
    runner()
        .run_test(&test)
        .expect("agg-count-distinct-01 failed");
}

#[test]
fn test_agg_count_distinct_02() {
    // SELECT (COUNT(DISTINCT ?name) AS ?count) WHERE { ?s foaf:name ?name }
    let ds = person_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            vec![],
            vec![(variable("count"), agg_count_distinct("name"))],
        ),
        vec![variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-count-distinct-02",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("count", "3")])]),
    );
    runner()
        .run_test(&test)
        .expect("agg-count-distinct-02 failed");
}

// ===== HAVING TESTS (modeled as post-filter) =====

#[test]
fn test_agg_having_01() {
    // SELECT ?cat (COUNT(*) AS ?count) WHERE { ?item :category ?cat } GROUP BY ?cat HAVING(?count > 2)
    // Category A has 3 items, B has 2 items; HAVING count > 2 => only A
    let ds = group_by_dataset();
    let group_algebra = group(
        bgp(vec![triple(var("item"), ex("category"), var("cat"))]),
        vec![group_var("cat")],
        vec![(variable("count"), agg_count_star())],
    );
    // Apply HAVING as a filter after group
    let algebra = project(
        filter(
            group_algebra,
            expr_gt(expr_var("count"), expr_lit(lit_int(2))),
        ),
        vec![variable("cat"), variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-having-01",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only category A (count=3)
    );
    runner().run_test(&test).expect("agg-having-01 failed");
}

#[test]
fn test_agg_having_02_sum() {
    // SELECT ?cat (SUM(?price) AS ?total) WHERE { ?item :category ?cat . ?item :price ?price }
    // GROUP BY ?cat HAVING(?total > 50)
    // Category A: 10+20+30=60, Category B: 15+25=40; HAVING > 50 => only A
    let ds = group_by_dataset();
    let group_algebra = group(
        bgp(vec![
            triple(var("item"), ex("category"), var("cat")),
            triple(var("item"), ex("price"), var("price")),
        ]),
        vec![group_var("cat")],
        vec![(variable("total"), agg_sum("price"))],
    );
    let algebra = project(
        filter(
            group_algebra,
            expr_gt(expr_var("total"), expr_lit(lit_int(50))),
        ),
        vec![variable("cat"), variable("total")],
    );
    let test = ConformanceTest::new(
        "agg-having-02",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only category A
    );
    runner().run_test(&test).expect("agg-having-02 failed");
}

#[test]
fn test_agg_having_03_none_match() {
    // HAVING count > 10 — no category has more than 10 items
    let ds = group_by_dataset();
    let group_algebra = group(
        bgp(vec![triple(var("item"), ex("category"), var("cat"))]),
        vec![group_var("cat")],
        vec![(variable("count"), agg_count_star())],
    );
    let algebra = project(
        filter(
            group_algebra,
            expr_gt(expr_var("count"), expr_lit(lit_int(10))),
        ),
        vec![variable("cat"), variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-having-03",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("agg-having-03 failed");
}

#[test]
fn test_agg_multiple_aggregates() {
    // SELECT (COUNT(*) AS ?count) (SUM(?v) AS ?sum) (AVG(?v) AS ?avg) WHERE { ?s :value ?v }
    let ds = numeric_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), ex("value"), var("v"))]),
            vec![],
            vec![
                (variable("count"), agg_count_star()),
                (variable("sum"), agg_sum("v")),
                (variable("avg"), agg_avg("v")),
            ],
        ),
        vec![variable("count"), variable("sum"), variable("avg")],
    );
    let test = ConformanceTest::new(
        "agg-multi-01",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[
            ("count", "5"),
            ("sum", "100"),
            ("avg", "20"),
        ])]),
    );
    runner().run_test(&test).expect("agg-multi-01 failed");
}

#[test]
fn test_agg_count_with_filter() {
    // SELECT (COUNT(*) AS ?count) WHERE { ?s foaf:age ?age . FILTER(?age > 25) }
    let ds = person_dataset();
    let algebra = project(
        group(
            filter(
                bgp(vec![triple(
                    var("s"),
                    iri(&format!("{FOAF}age")),
                    var("age"),
                )]),
                expr_gt(expr_var("age"), expr_lit(lit_int(25))),
            ),
            vec![],
            vec![(variable("count"), agg_count_star())],
        ),
        vec![variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-count-filter-01",
        ConformanceGroup::Aggregates,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("count", "2")])]), // alice(30), charlie(35)
    );
    runner()
        .run_test(&test)
        .expect("agg-count-filter-01 failed");
}

#[test]
fn test_agg_group_count_many_groups() {
    // SELECT ?s (COUNT(?p) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?s
    let ds = person_dataset();
    let algebra = project(
        group(
            bgp(vec![triple(var("s"), var("p"), var("o"))]),
            vec![group_var("s")],
            vec![(variable("count"), agg_count_star())],
        ),
        vec![variable("s"), variable("count")],
    );
    let test = ConformanceTest::new(
        "agg-group-count-many-01",
        ConformanceGroup::GroupBy,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // one per person
    );
    runner()
        .run_test(&test)
        .expect("agg-group-count-many-01 failed");
}
