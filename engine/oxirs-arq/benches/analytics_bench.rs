//! Criterion benchmarks for the graph-analytics aggregate module.
//!
//! Covers PageRank (100-node random graph), betweenness centrality (50-node),
//! and connected-component labelling (200-node).

use criterion::{criterion_group, criterion_main, Criterion};
use oxirs_arq::{DegreeDirection, GraphAnalyticsAccumulator, GraphAnalyticsAggregate};
use std::hint::black_box;

// ── Graph generators ──────────────────────────────────────────────────────────

/// Generate a deterministic "random" graph using a simple LCG.
/// Returns `(subject, object)` IRI pairs.
fn lcg_random_graph(nodes: usize, avg_degree: usize, seed: u64) -> Vec<(String, String)> {
    let mut state = seed;
    let mut edges = Vec::with_capacity(nodes * avg_degree);

    for src in 0..nodes {
        for _ in 0..avg_degree {
            // LCG: next = (a * state + c) mod m
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let dst = (state >> 33) as usize % nodes;
            if dst != src {
                edges.push((format!("ex:n{src}"), format!("ex:n{dst}")));
            }
        }
    }
    edges
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_pagerank(c: &mut Criterion) {
    let edges = lcg_random_graph(100, 3, 0xDEAD_BEEF);

    c.bench_function("pagerank_100_nodes", |b| {
        b.iter(|| {
            let mut acc = GraphAnalyticsAccumulator::new(GraphAnalyticsAggregate::PageRank {
                damping: 0.85,
                iterations: 50,
            });
            for (s, o) in &edges {
                acc.accumulate(black_box(s), black_box(o));
            }
            let scores = acc.finalize().expect("PageRank bench should succeed");
            black_box(scores)
        })
    });
}

fn bench_betweenness(c: &mut Criterion) {
    let edges = lcg_random_graph(50, 3, 0xCAFE_F00D);

    c.bench_function("betweenness_50_nodes", |b| {
        b.iter(|| {
            let mut acc =
                GraphAnalyticsAccumulator::new(GraphAnalyticsAggregate::BetweennessCentrality {
                    normalized: true,
                });
            for (s, o) in &edges {
                acc.accumulate(black_box(s), black_box(o));
            }
            let scores = acc.finalize().expect("betweenness bench should succeed");
            black_box(scores)
        })
    });
}

fn bench_components(c: &mut Criterion) {
    let edges = lcg_random_graph(200, 2, 0x1234_5678);

    c.bench_function("connected_components_200_nodes", |b| {
        b.iter(|| {
            let mut acc =
                GraphAnalyticsAccumulator::new(GraphAnalyticsAggregate::ConnectedComponent);
            for (s, o) in &edges {
                acc.accumulate(black_box(s), black_box(o));
            }
            let scores = acc.finalize().expect("components bench should succeed");
            black_box(scores)
        })
    });
}

fn bench_degree_centrality(c: &mut Criterion) {
    let edges = lcg_random_graph(150, 4, 0xABCD_EF01);

    c.bench_function("degree_centrality_150_nodes", |b| {
        b.iter(|| {
            let mut acc =
                GraphAnalyticsAccumulator::new(GraphAnalyticsAggregate::DegreeCentrality {
                    direction: DegreeDirection::Both,
                });
            for (s, o) in &edges {
                acc.accumulate(black_box(s), black_box(o));
            }
            let scores = acc
                .finalize()
                .expect("degree centrality bench should succeed");
            black_box(scores)
        })
    });
}

criterion_group!(
    benches,
    bench_pagerank,
    bench_betweenness,
    bench_components,
    bench_degree_centrality
);
criterion_main!(benches);
