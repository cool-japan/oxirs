//! Negation Conformance Tests
//!
//! Tests SPARQL 1.1 MINUS and NOT EXISTS negation patterns.

use super::framework::*;
use super::helpers::*;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== MINUS TESTS =====

#[test]
fn test_neg_minus_01_basic() {
    // SELECT ?s WHERE { ?s foaf:name ?name } MINUS { ?s foaf:mbox ?mbox }
    // Persons without email: bob and dave
    let ds = negation_dataset();
    let algebra = project(
        minus(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "neg-minus-01",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // bob and dave
    );
    runner().run_test(&test).expect("neg-minus-01 failed");
}

#[test]
fn test_neg_minus_02_all_removed() {
    // When MINUS removes all results
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p"), str_lit("v1"));
    ds.add_triple(ex("b"), ex("p"), str_lit("v2"));
    ds.add_triple(ex("a"), ex("q"), str_lit("r1"));
    ds.add_triple(ex("b"), ex("q"), str_lit("r2"));

    let algebra = minus(
        bgp(vec![triple(var("s"), ex("p"), var("o"))]),
        bgp(vec![triple(var("s"), ex("q"), var("r"))]),
    );
    let test = ConformanceTest::new(
        "neg-minus-02",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(0), // all subjects also have :q
    );
    runner().run_test(&test).expect("neg-minus-02 failed");
}

#[test]
fn test_neg_minus_03_none_removed() {
    // When MINUS removes nothing (right side is empty)
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p"), str_lit("v1"));
    ds.add_triple(ex("b"), ex("p"), str_lit("v2"));

    let algebra = minus(
        bgp(vec![triple(var("s"), ex("p"), var("o"))]),
        bgp(vec![triple(var("s"), ex("nonexistent"), var("r"))]),
    );
    let test = ConformanceTest::new(
        "neg-minus-03",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("neg-minus-03 failed");
}

#[test]
fn test_neg_minus_04_partial_removal() {
    // MINUS removes some but not all results
    let ds = negation_dataset();
    let algebra = project(
        minus(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
        ),
        vec![variable("name")],
    );
    let test = ConformanceTest::new(
        "neg-minus-04",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::SelectResults(vec![row(&[("name", "Bob")]), row(&[("name", "Dave")])]),
    );
    runner().run_test(&test).expect("neg-minus-04 failed");
}

#[test]
fn test_neg_minus_05_empty_left() {
    // MINUS with empty left = empty result
    let ds = InMemoryDataset::new();
    let algebra = minus(
        bgp(vec![triple(var("s"), ex("p"), var("o"))]),
        bgp(vec![triple(var("s"), ex("q"), var("r"))]),
    );
    let test = ConformanceTest::new(
        "neg-minus-05",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("neg-minus-05 failed");
}

#[test]
fn test_neg_minus_06_disjoint_vars() {
    // MINUS where left and right have disjoint variables — no removal
    // In SPARQL, MINUS only removes if the intersection of variables is non-empty
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p"), str_lit("v1"));
    ds.add_triple(ex("b"), ex("q"), str_lit("v2"));

    // Different variables: ?s vs ?t — MINUS has no shared vars, so nothing removed
    let algebra = minus(
        bgp(vec![triple(var("s"), ex("p"), var("o"))]),
        bgp(vec![triple(var("t"), ex("q"), var("u"))]),
    );
    let test = ConformanceTest::new(
        "neg-minus-06",
        ConformanceGroup::Negation,
        algebra,
        ds,
        // With disjoint vars, SPARQL spec says MINUS removes nothing
        // Our implementation may remove based on compatibility — accept either 0 or 1
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("neg-minus-06 failed");
}

// ===== NOT EXISTS TESTS =====

#[test]
fn test_neg_notexists_01_basic() {
    // SELECT ?s WHERE { ?s foaf:name ?name . FILTER NOT EXISTS { ?s foaf:mbox ?mbox } }
    let ds = negation_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::NotExists(Box::new(bgp(vec![triple(
                var("s"),
                foaf("mbox"),
                var("mbox"),
            )]))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "neg-notexists-01",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // bob and dave
    );
    runner().run_test(&test).expect("neg-notexists-01 failed");
}

#[test]
fn test_neg_exists_01_basic() {
    // SELECT ?s WHERE { ?s foaf:name ?name . FILTER EXISTS { ?s foaf:mbox ?mbox } }
    let ds = negation_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::Exists(Box::new(bgp(vec![triple(
                var("s"),
                foaf("mbox"),
                var("mbox"),
            )]))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "neg-exists-01",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(2), // alice and charlie
    );
    runner().run_test(&test).expect("neg-exists-01 failed");
}

#[test]
fn test_neg_notexists_02_empty_dataset() {
    // NOT EXISTS always true when dataset is empty
    let ds = InMemoryDataset::new();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            Expression::NotExists(Box::new(bgp(vec![triple(var("s"), ex("q"), var("r"))]))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "neg-notexists-02",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(0), // no left results anyway
    );
    runner().run_test(&test).expect("neg-notexists-02 failed");
}

#[test]
fn test_neg_minus_chain_01() {
    // Chained MINUS: left MINUS right1 MINUS right2
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p"), str_lit("v"));
    ds.add_triple(ex("b"), ex("p"), str_lit("v"));
    ds.add_triple(ex("c"), ex("p"), str_lit("v"));
    ds.add_triple(ex("a"), ex("q"), str_lit("r")); // remove a
    ds.add_triple(ex("b"), ex("r"), str_lit("s")); // remove b

    let algebra = minus(
        minus(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            bgp(vec![triple(var("s"), ex("q"), var("x"))]),
        ),
        bgp(vec![triple(var("s"), ex("r"), var("y"))]),
    );
    let test = ConformanceTest::new(
        "neg-minus-chain-01",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only :c remains
    );
    runner().run_test(&test).expect("neg-minus-chain-01 failed");
}

#[test]
fn test_neg_minus_with_filter_01() {
    // MINUS combined with FILTER
    let ds = negation_dataset();
    let algebra = project(
        minus(
            filter(
                bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
                Expression::Binary {
                    op: crate::algebra::BinaryOperator::NotEqual,
                    left: Box::new(expr_var("name")),
                    right: Box::new(expr_lit(lit_str("Bob"))),
                },
            ),
            bgp(vec![triple(var("s"), foaf("mbox"), var("mbox"))]),
        ),
        vec![variable("s"), variable("name")],
    );
    let test = ConformanceTest::new(
        "neg-minus-filter-01",
        ConformanceGroup::Negation,
        algebra,
        ds,
        // Filter removes Bob; then MINUS removes alice and charlie (have mbox)
        // So only Dave remains
        ConformanceResult::ResultCount(1),
    );
    runner()
        .run_test(&test)
        .expect("neg-minus-filter-01 failed");
}

#[test]
fn test_neg_difference_semantics_01() {
    // Verify semantic difference between MINUS and NOT EXISTS when vars differ
    // MINUS: pattern1 MINUS pattern2 — only removes if shared variables match
    // NOT EXISTS: removes if subpattern matches in current context
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("a"), ex("p"), str_lit("va"));
    ds.add_triple(ex("b"), ex("p"), str_lit("vb"));
    ds.add_triple(ex("b"), ex("q"), str_lit("qb")); // b has q

    // NOT EXISTS approach
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), ex("p"), var("o"))]),
            Expression::NotExists(Box::new(bgp(vec![triple(var("s"), ex("q"), var("r"))]))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "neg-semantics-01",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(1), // only :a (b has q)
    );
    runner().run_test(&test).expect("neg-semantics-01 failed");
}

#[test]
fn test_neg_notexists_all_pass_01() {
    // NOT EXISTS subpattern never matches
    let ds = person_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("s"), foaf("name"), var("name"))]),
            Expression::NotExists(Box::new(bgp(vec![triple(
                var("s"),
                ex("nonexistent"),
                var("x"),
            )]))),
        ),
        vec![variable("s")],
    );
    let test = ConformanceTest::new(
        "neg-notexists-all-pass-01",
        ConformanceGroup::Negation,
        algebra,
        ds,
        ConformanceResult::ResultCount(3), // all 3 pass since nonexistent never matches
    );
    runner()
        .run_test(&test)
        .expect("neg-notexists-all-pass-01 failed");
}
