//! Benchmarks for IndexedGraph performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_core::model::{Literal, NamedNode, Object, Predicate, Subject, Triple};
use oxirs_core::store::IndexedGraph;
use scirs2_core::random::{Rng, Random};

/// Helper to create a test triple
fn create_triple(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(
        NamedNode::new(s).unwrap(),
        NamedNode::new(p).unwrap(),
        Literal::new(o),
    )
}

/// Generate random test data
fn generate_test_triples(count: usize) -> Vec<Triple> {
    let mut random = Random::default();
    (0..count)
        .map(|_| {
            create_triple(
                &format!("http://example.org/s{}", random.gen_range(0..1000)),
                &format!("http://example.org/p{}", random.gen_range(0..10)),
                &format!("object{}", random.gen_range(0..10000)),
            )
        })
        .collect()
}

fn bench_single_insertions(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_insertions");

    for size in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let triples = generate_test_triples(size);
            b.iter(|| {
                let graph = IndexedGraph::new();
                for triple in &triples {
                    graph.insert(black_box(triple));
                }
            });
        });
    }

    group.finish();
}

fn bench_batch_insertions(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insertions");

    for size in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let triples = generate_test_triples(size);
            b.iter(|| {
                let graph = IndexedGraph::new();
                graph.batch_insert(black_box(&triples));
            });
        });
    }

    group.finish();
}

fn bench_query_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_patterns");

    // Setup graph with 10000 triples
    let graph = IndexedGraph::new();
    for i in 0..100 {
        for j in 0..10 {
            for k in 0..10 {
                let triple = create_triple(
                    &format!("http://example.org/s{i}"),
                    &format!("http://example.org/p{j}"),
                    &format!("object{k}"),
                );
                graph.insert(&triple);
            }
        }
    }

    // Benchmark subject queries
    group.bench_function("query_by_subject", |b| {
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/s50").unwrap());
        b.iter(|| graph.query(Some(black_box(&subject)), None, None));
    });

    // Benchmark predicate queries
    group.bench_function("query_by_predicate", |b| {
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p5").unwrap());
        b.iter(|| graph.query(None, Some(black_box(&predicate)), None));
    });

    // Benchmark object queries
    group.bench_function("query_by_object", |b| {
        let object = Object::Literal(Literal::new("object5"));
        b.iter(|| graph.query(None, None, Some(black_box(&object))));
    });

    // Benchmark SP queries
    group.bench_function("query_by_sp", |b| {
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/s50").unwrap());
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p5").unwrap());
        b.iter(|| graph.query(Some(black_box(&subject)), Some(black_box(&predicate)), None));
    });

    // Benchmark PO queries
    group.bench_function("query_by_po", |b| {
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p5").unwrap());
        let object = Object::Literal(Literal::new("object5"));
        b.iter(|| graph.query(None, Some(black_box(&predicate)), Some(black_box(&object))));
    });

    // Benchmark SO queries
    group.bench_function("query_by_so", |b| {
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/s50").unwrap());
        let object = Object::Literal(Literal::new("object5"));
        b.iter(|| graph.query(Some(black_box(&subject)), None, Some(black_box(&object))));
    });

    // Benchmark SPO queries (exact match)
    group.bench_function("query_by_spo", |b| {
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/s50").unwrap());
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p5").unwrap());
        let object = Object::Literal(Literal::new("object5"));
        b.iter(|| {
            graph.query(
                Some(black_box(&subject)),
                Some(black_box(&predicate)),
                Some(black_box(&object)),
            )
        });
    });

    // Benchmark all triples query
    group.bench_function("query_all", |b| {
        b.iter(|| graph.query(None, None, None));
    });

    group.finish();
}

fn bench_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("removal");

    group.bench_function("remove_existing", |b| {
        b.iter_with_setup(
            || {
                let graph = IndexedGraph::new();
                let triple =
                    create_triple("http://example.org/s", "http://example.org/p", "object");
                graph.insert(&triple);
                (graph, triple)
            },
            |(graph, triple)| graph.remove(black_box(&triple)),
        );
    });

    group.bench_function("remove_non_existing", |b| {
        let graph = IndexedGraph::new();
        let triple = create_triple(
            "http://example.org/nonexistent",
            "http://example.org/p",
            "object",
        );
        b.iter(|| graph.remove(black_box(&triple)));
    });

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    group.bench_function("memory_per_triple", |b| {
        b.iter_with_setup(IndexedGraph::new, |graph| {
            // Insert 1000 triples
            for i in 0..1000 {
                let triple = create_triple(
                    &format!("http://example.org/subject{i}"),
                    &format!("http://example.org/predicate{}", i % 50),
                    &format!("This is object number {i} with some text"),
                );
                graph.insert(&triple);
            }
            let usage = graph.memory_usage();
            (usage.total_bytes(), usage.bytes_per_triple())
        });
    });

    group.finish();
}

fn bench_concurrent_reads(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let mut group = c.benchmark_group("concurrent_reads");

    // Setup shared graph
    let graph = Arc::new(IndexedGraph::new());
    for i in 0..1000 {
        let triple = create_triple(
            &format!("http://example.org/s{}", i % 100),
            &format!("http://example.org/p{}", i % 10),
            &format!("object{i}"),
        );
        graph.insert(&triple);
    }

    group.bench_function("4_threads", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    let graph = Arc::clone(&graph);
                    thread::spawn(move || {
                        let subject = Subject::NamedNode(
                            NamedNode::new(format!("http://example.org/s{}", i * 25)).unwrap(),
                        );
                        graph.query(Some(&subject), None, None).len()
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .sum::<usize>()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_insertions,
    bench_batch_insertions,
    bench_query_patterns,
    bench_removal,
    bench_memory_efficiency,
    bench_concurrent_reads
);
criterion_main!(benches);
