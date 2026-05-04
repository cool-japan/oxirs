//! Integration tests for the graph-analytics aggregate module.
//!
//! These tests exercise `GraphAnalyticsAccumulator` end-to-end through the
//! public crate API, covering all five analytics variants and the accumulator
//! mechanics.

use std::collections::HashSet;

use oxirs_arq::{
    DegreeDirection, GraphAnalyticsAccumulator, GraphAnalyticsAggregate, GraphAnalyticsError,
};

// ── Helper ────────────────────────────────────────────────────────────────────

fn make_acc(kind: GraphAnalyticsAggregate) -> GraphAnalyticsAccumulator {
    GraphAnalyticsAccumulator::new(kind)
}

// ── PageRank ──────────────────────────────────────────────────────────────────

#[test]
fn test_pagerank_4node_cycle_converges() {
    let mut acc = make_acc(GraphAnalyticsAggregate::PageRank {
        damping: 0.85,
        iterations: 100,
    });
    // 4-cycle: A→B→C→D→A
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:B", "ex:C");
    acc.accumulate("ex:C", "ex:D");
    acc.accumulate("ex:D", "ex:A");

    let scores = acc.finalize().expect("PageRank on 4-cycle should succeed");
    assert_eq!(scores.len(), 4, "4 distinct nodes expected");

    // In a symmetric directed cycle all nodes have equal rank.
    let expected = 1.0 / 4.0;
    for (node, &score) in &scores {
        assert!(
            (score - expected).abs() < 1e-4,
            "node {node}: expected ~{expected}, got {score}"
        );
    }

    // Scores must sum to 1.
    let total: f64 = scores.values().sum();
    assert!((total - 1.0).abs() < 1e-4, "score sum={total}");
}

#[test]
fn test_pagerank_star_center_highest() {
    let mut acc = make_acc(GraphAnalyticsAggregate::PageRank {
        damping: 0.85,
        iterations: 100,
    });
    acc.accumulate("ex:S1", "ex:Hub");
    acc.accumulate("ex:S2", "ex:Hub");
    acc.accumulate("ex:S3", "ex:Hub");
    // Hub links back to avoid dangling
    acc.accumulate("ex:Hub", "ex:S1");

    let scores = acc.finalize().expect("star PageRank should succeed");
    let hub = scores["ex:Hub"];
    let s2 = scores["ex:S2"];
    assert!(hub > s2, "hub PageRank ({hub}) should exceed spoke ({s2})");
}

#[test]
fn test_pagerank_empty_graph_error() {
    let mut acc = make_acc(GraphAnalyticsAggregate::PageRank {
        damping: 0.85,
        iterations: 100,
    });
    let err = acc.finalize().expect_err("empty graph must return Err");
    assert!(
        matches!(err, GraphAnalyticsError::EmptyGraph),
        "expected EmptyGraph, got: {err}"
    );
}

#[test]
fn test_pagerank_damping_affects_rank() {
    // Different damping factors must both produce valid (sum-to-one) rank
    // distributions even though the per-node values differ.
    let edges = [
        ("ex:A", "ex:B"),
        ("ex:B", "ex:C"),
        ("ex:C", "ex:A"),
        ("ex:D", "ex:A"),
    ];

    let mut acc_low = make_acc(GraphAnalyticsAggregate::PageRank {
        damping: 0.3,
        iterations: 200,
    });
    let mut acc_high = make_acc(GraphAnalyticsAggregate::PageRank {
        damping: 0.95,
        iterations: 200,
    });

    for (s, o) in &edges {
        acc_low.accumulate(s, o);
        acc_high.accumulate(s, o);
    }

    let s_low = acc_low.finalize().expect("low damping should succeed");
    let s_high = acc_high.finalize().expect("high damping should succeed");

    let sum_low: f64 = s_low.values().sum();
    let sum_high: f64 = s_high.values().sum();

    assert!(
        (sum_low - 1.0).abs() < 1e-4,
        "low-damping sum={sum_low} must be ~1"
    );
    assert!(
        (sum_high - 1.0).abs() < 1e-4,
        "high-damping sum={sum_high} must be ~1"
    );

    // The two distributions should differ (different damping)
    let diff: f64 = s_low.iter().map(|(k, &v)| (v - s_high[k]).abs()).sum();
    assert!(
        diff > 1e-6,
        "damping factors should produce different ranks"
    );
}

// ── Betweenness centrality ────────────────────────────────────────────────────

#[test]
fn test_betweenness_bridge_node_highest() {
    // A → B → C  — B is the only bridge
    let mut acc = make_acc(GraphAnalyticsAggregate::BetweennessCentrality { normalized: true });
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:B", "ex:C");

    let scores = acc.finalize().expect("betweenness should succeed");
    let b = scores["ex:B"];
    let a = scores["ex:A"];
    let c = scores["ex:C"];

    assert!(b > a, "bridge B ({b}) > A ({a})");
    assert!(b > c, "bridge B ({b}) > C ({c})");
}

#[test]
fn test_betweenness_complete_graph_equal() {
    // Directed K3: all nodes have equal betweenness
    let mut acc = make_acc(GraphAnalyticsAggregate::BetweennessCentrality { normalized: true });
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:A", "ex:C");
    acc.accumulate("ex:B", "ex:A");
    acc.accumulate("ex:B", "ex:C");
    acc.accumulate("ex:C", "ex:A");
    acc.accumulate("ex:C", "ex:B");

    let scores = acc.finalize().expect("K3 betweenness should succeed");
    let vals: Vec<f64> = scores.values().copied().collect();
    let first = vals[0];
    for &v in &vals {
        assert!(
            (v - first).abs() < 1e-9,
            "K3: all betweenness equal; {v} vs {first}"
        );
    }
}

#[test]
fn test_betweenness_normalized_in_range() {
    let mut acc = make_acc(GraphAnalyticsAggregate::BetweennessCentrality { normalized: true });
    // 5-node chain
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:B", "ex:C");
    acc.accumulate("ex:C", "ex:D");
    acc.accumulate("ex:D", "ex:E");

    let scores = acc.finalize().expect("chain betweenness should succeed");
    for (node, &score) in &scores {
        assert!(
            (0.0..=1.0).contains(&score),
            "node {node} score {score} out of [0,1]"
        );
    }
}

// ── Connected components ──────────────────────────────────────────────────────

#[test]
fn test_connected_component_single_component() {
    let mut acc = make_acc(GraphAnalyticsAggregate::ConnectedComponent);
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:B", "ex:C");
    acc.accumulate("ex:C", "ex:A");

    let scores = acc.finalize().expect("single component should succeed");
    assert_eq!(scores.len(), 3);
    // All nodes in the same component
    let ids: HashSet<usize> = scores.values().map(|&v| v as usize).collect();
    assert_eq!(ids.len(), 1, "all nodes must share one component ID");
}

#[test]
fn test_connected_component_two_components() {
    let mut acc = make_acc(GraphAnalyticsAggregate::ConnectedComponent);
    acc.accumulate("ex:A", "ex:B"); // component X
    acc.accumulate("ex:C", "ex:D"); // component Y

    let scores = acc.finalize().expect("two-component graph should succeed");
    let ids: HashSet<usize> = scores.values().map(|&v| v as usize).collect();
    assert_eq!(ids.len(), 2, "exactly 2 component IDs expected");
}

#[test]
fn test_connected_component_isolated_node() {
    // {A, B} and {C, D} are isolated from each other — similar to two-component
    // but checking the isolated pair explicitly.
    let mut acc = make_acc(GraphAnalyticsAggregate::ConnectedComponent);
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:C", "ex:D");

    let scores = acc.finalize().expect("isolated node test should succeed");
    assert_eq!(scores.len(), 4);

    // A and B must share a component; C and D must share a different one.
    let id_a = scores["ex:A"] as usize;
    let id_b = scores["ex:B"] as usize;
    let id_c = scores["ex:C"] as usize;
    let id_d = scores["ex:D"] as usize;

    assert_eq!(id_a, id_b, "A and B must be in the same component");
    assert_eq!(id_c, id_d, "C and D must be in the same component");
    assert_ne!(id_a, id_c, "the two pairs must be in different components");
}

// ── Clustering coefficient ────────────────────────────────────────────────────

#[test]
fn test_clustering_complete_graph_is_one() {
    // Triangle K3 (directed edges forming undirected K3)
    let mut acc = make_acc(GraphAnalyticsAggregate::ClusteringCoefficient);
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:B", "ex:C");
    acc.accumulate("ex:A", "ex:C");

    let scores = acc.finalize().expect("K3 clustering should succeed");
    for (node, &coeff) in &scores {
        assert!(
            (coeff - 1.0).abs() < 1e-9,
            "K3 node {node}: expected 1.0, got {coeff}"
        );
    }
}

#[test]
fn test_clustering_star_is_zero() {
    // Star: hub → S1, S2, S3 — spokes are not connected to each other
    let mut acc = make_acc(GraphAnalyticsAggregate::ClusteringCoefficient);
    acc.accumulate("ex:Hub", "ex:S1");
    acc.accumulate("ex:Hub", "ex:S2");
    acc.accumulate("ex:Hub", "ex:S3");

    let scores = acc.finalize().expect("star clustering should succeed");
    for spoke in &["ex:S1", "ex:S2", "ex:S3"] {
        let coeff = scores[*spoke];
        assert!(coeff < 1e-9, "spoke {spoke}: expected 0.0, got {coeff}");
    }
}

// ── Degree centrality ─────────────────────────────────────────────────────────

#[test]
fn test_degree_outgoing_count() {
    // A→B, A→C, B→C: A has out-degree 2, B has 1, C has 0
    let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
        direction: DegreeDirection::Outgoing,
    });
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:A", "ex:C");
    acc.accumulate("ex:B", "ex:C");

    let scores = acc.finalize().expect("outgoing degree should succeed");
    let a = scores["ex:A"];
    let b = scores["ex:B"];
    let c = scores["ex:C"];

    // n=3, norm=n-1=2; A:2/2=1.0, B:1/2=0.5, C:0/2=0.0
    assert!((a - 1.0).abs() < 1e-9, "A out={a}");
    assert!((b - 0.5).abs() < 1e-9, "B out={b}");
    assert!((c - 0.0).abs() < 1e-9, "C out={c}");
}

#[test]
fn test_degree_incoming_count() {
    // A→C, B→C, A→B: C has in=2, B has in=1, A has in=0
    let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
        direction: DegreeDirection::Incoming,
    });
    acc.accumulate("ex:A", "ex:C");
    acc.accumulate("ex:B", "ex:C");
    acc.accumulate("ex:A", "ex:B");

    let scores = acc.finalize().expect("incoming degree should succeed");
    let c = scores["ex:C"];
    let b = scores["ex:B"];
    let a = scores["ex:A"];

    // n=3, norm=2; C:2/2=1.0, B:1/2=0.5, A:0/2=0.0
    assert!((c - 1.0).abs() < 1e-9, "C in={c}");
    assert!((b - 0.5).abs() < 1e-9, "B in={b}");
    assert!((a - 0.0).abs() < 1e-9, "A in={a}");
}

#[test]
fn test_degree_both_sum() {
    // A→B, B→C: A:out=1,in=0; B:out=1,in=1; C:out=0,in=1
    let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
        direction: DegreeDirection::Both,
    });
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:B", "ex:C");

    let scores = acc.finalize().expect("both-degree should succeed");
    let b = scores["ex:B"];
    let a = scores["ex:A"];
    let c = scores["ex:C"];

    // B has highest total degree
    assert!(b > a, "B total ({b}) > A total ({a})");
    assert!(b > c, "B total ({b}) > C total ({c})");
    // A and C are symmetric (same total degree)
    assert!((a - c).abs() < 1e-9, "A ({a}) == C ({c}) total degree");
}

// ── Accumulator mechanics ─────────────────────────────────────────────────────

#[test]
fn test_accumulate_tracks_edge_count() {
    let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
        direction: DegreeDirection::Both,
    });
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:B", "ex:C");
    acc.accumulate("ex:C", "ex:D");
    assert_eq!(acc.edge_count(), 3);
    assert_eq!(acc.node_count(), 4);
}

#[test]
fn test_finalize_for_node_existing() {
    // ex:A has out-degree 2, n=3, normalised score = 2/(3-1) = 1.0
    let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
        direction: DegreeDirection::Outgoing,
    });
    acc.accumulate("ex:A", "ex:B");
    acc.accumulate("ex:A", "ex:C");

    let score = acc
        .finalize_for_node("ex:A")
        .expect("finalize should succeed")
        .expect("ex:A must have a score");

    assert!((score - 1.0).abs() < 1e-9, "ex:A out-degree score={score}");
}

#[test]
fn test_finalize_for_node_missing_returns_none() {
    let mut acc = make_acc(GraphAnalyticsAggregate::DegreeCentrality {
        direction: DegreeDirection::Both,
    });
    acc.accumulate("ex:A", "ex:B");

    let result = acc
        .finalize_for_node("ex:Z")
        .expect("finalize should succeed");

    assert!(
        result.is_none(),
        "ex:Z was never accumulated; expected None"
    );
}
