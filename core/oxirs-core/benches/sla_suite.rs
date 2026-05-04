//! SLA benchmark suite for oxirs-core.
//!
//! Benchmarks in this file correspond to named SLO targets defined in
//! `core/oxirs-core/perf_baseline.json`. Run with:
//!
//! ```text
//! cargo bench -p oxirs-core --bench sla_suite
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_core::model::NamedNode;

/// Benchmark: equality comparison of two identical [`NamedNode`] values.
///
/// SLO target: p50 < 100µs (see `perf_baseline.json`).
fn bench_term_equality(c: &mut Criterion) {
    let a = NamedNode::new("https://example.org/subject").expect("valid IRI");
    let b = NamedNode::new("https://example.org/subject").expect("valid IRI");

    c.bench_function("sla_term_equality", |bencher| {
        bencher.iter(|| std::hint::black_box(&a) == std::hint::black_box(&b))
    });
}

/// Benchmark: counting lines of a pre-built N-Triples payload.
///
/// This exercises cheap string scanning as a proxy for parsing throughput.
/// Real parser benchmarks live in `rdf_bench.rs`; this one verifies that
/// even the most trivial per-line overhead does not accumulate unexpectedly.
fn bench_ntriples_line_count(c: &mut Criterion) {
    // Generate 1 000 synthetic N-Triples lines entirely in memory — no I/O.
    let data: String = (0..1_000_u32)
        .map(|i| {
            format!(
                "<https://example.org/s{i}> <https://example.org/p> <https://example.org/o{i}> .\n"
            )
        })
        .collect();

    c.bench_function("sla_ntriples_line_count_1k", |bencher| {
        bencher.iter(|| {
            let count = std::hint::black_box(data.as_str()).lines().count();
            std::hint::black_box(count)
        })
    });
}

/// Benchmark: allocating and comparing [`NamedNode`] instances at different
/// IRI lengths to detect quadratic or otherwise unexpected scaling.
fn bench_named_node_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sla_named_node_scaling");
    for len in [16_usize, 64, 256, 1024] {
        let iri = format!("https://example.org/{}", "x".repeat(len));
        group.bench_with_input(BenchmarkId::from_parameter(len), &iri, |bencher, iri| {
            bencher.iter(|| NamedNode::new(std::hint::black_box(iri.as_str())).expect("valid IRI"))
        });
    }
    group.finish();
}

criterion_group!(
    sla_benches,
    bench_term_equality,
    bench_ntriples_line_count,
    bench_named_node_scaling,
);
criterion_main!(sla_benches);
