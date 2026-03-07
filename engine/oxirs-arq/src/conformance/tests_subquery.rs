//! Subquery Conformance Tests
//!
//! Tests SPARQL 1.1 subquery patterns (SELECT within SELECT).

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== BASIC SUBQUERY TESTS =====

#[test]
fn test_subq_01_basic() {
    // SELECT ?name WHERE {
    //   { SELECT ?s WHERE { ?s :score ?sc . FILTER(?sc = 92) } }
    //   ?s foaf:name ?name
    // }
    let ds = subquery_dataset();

    // Inner subquery: persons with score = 92
    let inner = project(
        filter(
            bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
            expr_eq(expr_var("sc"), expr_lit(lit_int(92))),
        ),
        vec![variable("s")],
    );

    // Outer query: get names
    let outer_bgp = bgp(vec![triple(var("s"), foaf("name"), var("name"))]);
    let algebra = project(join(inner, outer_bgp), vec![variable("name")]);

    let test = ConformanceTest::new(
        "subq-01",
        ConformanceGroup::Subquery,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // bob, dave (both have score 92)
    );
    runner().run_test(&test).expect("subq-01 failed");
}

#[test]
fn test_subq_02_aggregate_in_subquery() {
    // SELECT ?name WHERE {
    //   { SELECT (MAX(?sc) AS ?max_score) WHERE { ?s :score ?sc } }
    //   ?s :score ?score . ?s foaf:name ?name
    //   FILTER(?score = ?max_score)
    // }
    let ds = subquery_dataset();

    let inner = project(
        group(
            bgp(vec![triple(var("s_inner"), ex("score"), var("sc"))]),
            vec![],
            vec![(variable("max_score"), agg_max("sc"))],
        ),
        vec![variable("max_score")],
    );

    let outer = filter(
        join(
            inner,
            join(
                bgp(vec![triple(var("s"), ex("score"), var("score"))]),
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            ),
        ),
        expr_eq(expr_var("score"), expr_var("max_score")),
    );

    let algebra = project(outer, vec![variable("name")]);

    let test = ConformanceTest::new(
        "subq-02",
        ConformanceGroup::Subquery,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // bob and dave (both score 92)
    );
    runner().run_test(&test).expect("subq-02 failed");
}

#[test]
fn test_subq_03_limit_in_subquery() {
    // SELECT ?name WHERE {
    //   { SELECT ?s WHERE { ?s :score ?sc } LIMIT 2 }
    //   ?s foaf:name ?name
    // }
    let ds = subquery_dataset();

    let inner = slice(
        project(
            bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
            vec![variable("s")],
        ),
        None,
        Some(2),
    );

    let algebra = project(
        join(
            inner,
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        ),
        vec![variable("name")],
    );

    let test = ConformanceTest::new(
        "subq-03",
        ConformanceGroup::Subquery,
        algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("subq-03 failed");
}

#[test]
fn test_subq_04_order_in_subquery() {
    // SELECT ?name WHERE {
    //   { SELECT ?s WHERE { ?s :score ?sc } ORDER BY DESC(?sc) LIMIT 1 }
    //   ?s foaf:name ?name
    // }
    let ds = subquery_dataset();

    let inner = slice(
        order_by(
            project(
                bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
                vec![variable("s"), variable("sc")],
            ),
            vec![desc_cond(expr_var("sc"))],
        ),
        None,
        Some(1),
    );

    let algebra = project(
        join(
            inner,
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        ),
        vec![variable("name")],
    );

    let test = ConformanceTest::new(
        "subq-04",
        ConformanceGroup::Subquery,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("subq-04 failed");
}

#[test]
fn test_subq_05_count_in_subquery() {
    // SELECT ?count WHERE { { SELECT (COUNT(*) AS ?count) WHERE { ?s :score ?sc } } }
    let ds = subquery_dataset();

    let inner = project(
        group(
            bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
            vec![],
            vec![(variable("count"), agg_count_star())],
        ),
        vec![variable("count")],
    );

    let test = ConformanceTest::new(
        "subq-05",
        ConformanceGroup::Subquery,
        inner,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("count", "4")])]),
    );
    runner().run_test(&test).expect("subq-05 failed");
}

#[test]
fn test_subq_06_group_in_subquery() {
    // SELECT ?s ?avg_score WHERE {
    //   { SELECT ?s (AVG(?sc) AS ?avg_score) WHERE { ?s :score ?sc } GROUP BY ?s }
    // }
    let ds = subquery_dataset();

    let inner = project(
        group(
            bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
            vec![group_var("s")],
            vec![(variable("avg_score"), agg_avg("sc"))],
        ),
        vec![variable("s"), variable("avg_score")],
    );

    let test = ConformanceTest::new(
        "subq-06",
        ConformanceGroup::Subquery,
        inner,
        ds,
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("subq-06 failed");
}

#[test]
fn test_subq_07_empty_subquery() {
    // SELECT ?name WHERE {
    //   { SELECT ?s WHERE { ?s :nonexistent ?o } }
    //   ?s foaf:name ?name
    // }
    let ds = subquery_dataset();

    let inner = project(
        bgp(vec![triple(var("s"), ex("nonexistent"), var("o"))]),
        vec![variable("s")],
    );

    let algebra = project(
        join(
            inner,
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        ),
        vec![variable("name")],
    );

    let test = ConformanceTest::new(
        "subq-07",
        ConformanceGroup::Subquery,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("subq-07 failed");
}

#[test]
fn test_subq_08_nested_subqueries() {
    // Nested subqueries
    let ds = subquery_dataset();

    let innermost = project(
        bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
        vec![variable("s"), variable("sc")],
    );

    let middle = project(
        filter(innermost, expr_gt(expr_var("sc"), expr_lit(lit_int(89)))),
        vec![variable("s")],
    );

    let algebra = project(
        join(
            middle,
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
        ),
        vec![variable("name")],
    );

    let test = ConformanceTest::new(
        "subq-08",
        ConformanceGroup::Subquery,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // bob, dave
    );
    runner().run_test(&test).expect("subq-08 failed");
}

#[test]
fn test_subq_09_distinct_in_subquery() {
    // SELECT DISTINCT ?s WHERE { ?s :score ?sc }
    let ds = subquery_dataset();

    let inner = distinct(project(
        bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
        vec![variable("s")],
    ));

    let test = ConformanceTest::new(
        "subq-09",
        ConformanceGroup::Subquery,
        inner,
        ds,
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("subq-09 failed");
}

#[test]
fn test_subq_10_union_in_subquery() {
    // SELECT DISTINCT ?s WHERE { { ?s :score ?sc } UNION { ?s foaf:name ?n } }
    let ds = subquery_dataset();

    let inner = distinct(project(
        union(
            bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
            bgp(vec![triple(var("s"), foaf("name"), var("n"))]),
        ),
        vec![variable("s")],
    ));

    let test = ConformanceTest::new(
        "subq-10",
        ConformanceGroup::Subquery,
        inner,
        ds,
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("subq-10 failed");
}

#[test]
fn test_subq_11_filter_after_subquery() {
    // SELECT ?name ?sc WHERE {
    //   { SELECT ?s ?sc WHERE { ?s :score ?sc } }
    //   ?s foaf:name ?name
    //   FILTER(?sc < 85)
    // }
    let ds = subquery_dataset();

    let inner = project(
        bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
        vec![variable("s"), variable("sc")],
    );

    let algebra = project(
        filter(
            join(
                inner,
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            ),
            expr_lt(expr_var("sc"), expr_lit(lit_int(85))),
        ),
        vec![variable("name"), variable("sc")],
    );

    let test = ConformanceTest::new(
        "subq-11",
        ConformanceGroup::Subquery,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only charlie (78)
    );
    runner().run_test(&test).expect("subq-11 failed");
}

#[test]
fn test_subq_12_min_score() {
    // SELECT (MIN(?sc) AS ?min) WHERE { ?s :score ?sc }
    let ds = subquery_dataset();

    let algebra = project(
        group(
            bgp(vec![triple(var("s"), ex("score"), var("sc"))]),
            vec![],
            vec![(variable("min"), agg_min("sc"))],
        ),
        vec![variable("min")],
    );

    let test = ConformanceTest::new(
        "subq-12",
        ConformanceGroup::Subquery,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("min", "78")])]),
    );
    runner().run_test(&test).expect("subq-12 failed");
}
