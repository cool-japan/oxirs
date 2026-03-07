//! Property Path Conformance Tests
//!
//! Tests SPARQL 1.1 property paths: sequence, alternative, zero-or-more,
//! one-or-more, zero-or-one, inverse, and negated property sets.

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== SEQUENCE PATH TESTS =====

#[test]
fn test_pp_01_sequence_direct() {
    // ?s ex:child/ex:child ?o  (2-hop path)
    let ds = hierarchy_dataset();
    let path = path_seq(
        path_iri(&format!("{EX}child")),
        path_iri(&format!("{EX}child")),
    );
    let algebra = property_path(var("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-01",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // a->c, b->d, c->e
    );
    runner().run_test(&test).expect("pp-01 failed");
}

#[test]
fn test_pp_02_sequence_triple_hop() {
    // ?s ex:child/ex:child/ex:child ?o  (3-hop path)
    let ds = hierarchy_dataset();
    let path = path_seq(
        path_seq(
            path_iri(&format!("{EX}child")),
            path_iri(&format!("{EX}child")),
        ),
        path_iri(&format!("{EX}child")),
    );
    let algebra = property_path(var("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-02",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // a->d, b->e
    );
    runner().run_test(&test).expect("pp-02 failed");
}

// ===== ALTERNATIVE PATH TESTS =====

#[test]
fn test_pp_03_alternative() {
    // ?s ex:child|ex:knows ?o
    let ds = hierarchy_dataset();
    let path = path_alt(
        path_iri(&format!("{EX}child")),
        path_iri(&format!("{EX}knows")),
    );
    let algebra = property_path(var("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-03",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // Alternative path deduplicates: (a,b), (b,c), (c,d), (d,e) = 4 unique (s,o) pairs
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("pp-03 failed");
}

#[test]
fn test_pp_04_alternative_three() {
    // Three alternative predicates
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("s"), ex("p1"), ex("o1"));
    ds.add_triple(ex("s"), ex("p2"), ex("o2"));
    ds.add_triple(ex("s"), ex("p3"), ex("o3"));
    ds.add_triple(ex("s"), ex("p4"), ex("o4")); // not in alt

    let path = path_alt(
        path_alt(path_iri(&format!("{EX}p1")), path_iri(&format!("{EX}p2"))),
        path_iri(&format!("{EX}p3")),
    );
    let algebra = property_path(ex("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-04",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // o1, o2, o3
    );
    runner().run_test(&test).expect("pp-04 failed");
}

// ===== ZERO OR MORE TESTS =====

#[test]
fn test_pp_05_zero_or_more_from_specific() {
    // :a ex:child* ?o
    let ds = hierarchy_dataset();
    let path = path_star(path_iri(&format!("{EX}child")));
    let algebra = property_path(ex("a"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-05",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // :a ex:child* reachable: a(0 hops), b(1), c(2), d(3), e(4) = 5
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("pp-05 failed");
}

#[test]
fn test_pp_06_zero_or_more_variable_subject() {
    // ?s ex:child* :e  (who can reach :e via child*)
    let ds = hierarchy_dataset();
    let path = path_star(path_iri(&format!("{EX}child")));
    let algebra = property_path(var("s"), path, ex("e"));
    let test = ConformanceTest::new(
        "pp-06",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // All of a,b,c,d,e can reach e via child*
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("pp-06 failed");
}

// ===== ONE OR MORE TESTS =====

#[test]
fn test_pp_07_one_or_more() {
    // :a ex:child+ ?o  (1 or more hops)
    let ds = hierarchy_dataset();
    let path = path_plus(path_iri(&format!("{EX}child")));
    let algebra = property_path(ex("a"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-07",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // :a ex:child+ reachable: b(1), c(2), d(3), e(4) = 4 (NOT a itself)
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("pp-07 failed");
}

#[test]
fn test_pp_08_one_or_more_direct() {
    // Simple one-or-more with single hop available
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p"), ex("b"));

    let path = path_plus(path_iri(&format!("{EX}p")));
    let algebra = property_path(ex("a"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-08",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only b
    );
    runner().run_test(&test).expect("pp-08 failed");
}

// ===== ZERO OR ONE TESTS =====

#[test]
fn test_pp_09_zero_or_one_present() {
    // :a ex:child? ?o  (0 or 1 hops)
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("child"), ex("b"));

    let path = path_opt(path_iri(&format!("{EX}child")));
    let algebra = property_path(ex("a"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-09",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // 0 hops: a itself; 1 hop: b => 2 results
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("pp-09 failed");
}

#[test]
fn test_pp_10_zero_or_one_absent() {
    // :a ex:child? ?o when no child exists (only zero hop)
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("other"), ex("c"));

    let path = path_opt(path_iri(&format!("{EX}child")));
    let algebra = property_path(ex("a"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-10",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // 0 hops: a itself only
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("pp-10 failed");
}

// ===== INVERSE PATH TESTS =====

#[test]
fn test_pp_11_inverse() {
    // ?s ^ex:child :b  (i.e., :b ex:child ?s => :a)
    let ds = hierarchy_dataset();
    let path = path_inv(path_iri(&format!("{EX}child")));
    let algebra = property_path(var("s"), path, ex("b"));
    let test = ConformanceTest::new(
        "pp-11",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only :a has child :b
    );
    runner().run_test(&test).expect("pp-11 failed");
}

#[test]
fn test_pp_12_inverse_full() {
    // ?s ^ex:child ?o  (all inverse child relationships)
    let ds = hierarchy_dataset();
    let path = path_inv(path_iri(&format!("{EX}child")));
    let algebra = property_path(var("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-12",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        ConformanceResult::ResultCount(4), // b-a, c-b, d-c, e-d
    );
    runner().run_test(&test).expect("pp-12 failed");
}

// ===== NEGATED PROPERTY SET =====

#[test]
fn test_pp_13_negated() {
    // ?s !(ex:child) ?o  (any predicate except ex:child)
    let ds = hierarchy_dataset();
    let path = path_neg(vec![path_iri(&format!("{EX}child"))]);
    let algebra = property_path(var("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-13",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // only ex:knows predicates
    );
    runner().run_test(&test).expect("pp-13 failed");
}

#[test]
fn test_pp_14_negated_empty_set() {
    // ?s !() ?o  (any predicate — negated empty set)
    let ds = hierarchy_dataset();
    let path = path_neg(vec![]);
    let algebra = property_path(var("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-14",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // Per (s,o) pairs (deduplicated): (a,b), (b,c), (c,d), (d,e) = 4
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("pp-14 failed");
}

// ===== COMPLEX PATH COMBINATIONS =====

#[test]
fn test_pp_15_seq_star() {
    // ?s (ex:knows/ex:child*) ?o
    let ds = hierarchy_dataset();
    let path = path_seq(
        path_iri(&format!("{EX}knows")),
        path_star(path_iri(&format!("{EX}child"))),
    );
    let algebra = property_path(var("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-15",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // knows: a->b, b->c
        // then child* from b: b,c,d,e => a knows-then-child* b,c,d,e
        // then child* from c: c,d,e => b knows-then-child* c,d,e
        ConformanceResult::ResultCount(7), // a:{b,c,d,e}=4, b:{c,d,e}=3 => 7
    );
    runner().run_test(&test).expect("pp-15 failed");
}

#[test]
fn test_pp_16_alt_star() {
    // ?s (ex:child|ex:knows)* ?o from :a
    let ds = hierarchy_dataset();
    let path = path_star(path_alt(
        path_iri(&format!("{EX}child")),
        path_iri(&format!("{EX}knows")),
    ));
    let algebra = property_path(ex("a"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-16",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // All nodes reachable from a via any combination of child/knows
        // a(0 hops), b(a knows b or a child b), c, d, e
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("pp-16 failed");
}

// ===== SIMPLE PATH ON TYPED DATA =====

#[test]
fn test_pp_17_direct_iri_path() {
    // Simple direct IRI path — should behave like regular triple pattern
    let ds = person_dataset();
    let path = path_iri(&format!("{FOAF}name"));
    let algebra = property_path(var("s"), path, var("o"));
    let test = ConformanceTest::new(
        "pp-17",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("pp-17 failed");
}

#[test]
fn test_pp_18_path_with_bgp_join() {
    // Join property path with BGP
    let ds = hierarchy_dataset();
    let path_algebra = property_path(
        var("s"),
        path_plus(path_iri(&format!("{EX}child"))),
        var("o"),
    );
    let bgp_algebra = bgp(vec![triple(var("o"), ex("child"), var("leaf"))]);
    let algebra = project(
        join(path_algebra, bgp_algebra),
        vec![variable("s"), variable("o"), variable("leaf")],
    );
    let test = ConformanceTest::new(
        "pp-18",
        ConformanceGroup::PropertyPaths,
        algebra,
        ds,
        // s -> o (via child+), then o -> leaf (via child)
        // e.g. a->b, b->c; a->c, c->d; a->d, d->e; b->c, c->d; b->d, d->e; c->d, d->e
        ConformanceResult::ResultCount(6),
    );
    runner().run_test(&test).expect("pp-18 failed");
}
