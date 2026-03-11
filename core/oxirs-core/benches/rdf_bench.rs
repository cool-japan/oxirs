//! Core RDF Operations Benchmarks for oxirs-core
//!
//! v1.0.0 LTS benchmark suite covering:
//! - Term creation: IRIs, literals, blank nodes
//! - Triple construction
//! - In-memory graph insert
//! - Triple pattern matching
//! - IRI string parsing

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_core::{
    model::{BlankNode, Literal, NamedNode, Object, Predicate, Quad, Subject, Triple},
    rdf_store::ConcreteStore,
    store::IndexedGraph,
};
use std::hint::black_box;
use std::time::Duration;

// --- Term creation benchmarks ---

fn bench_term_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf/term_creation");
    group.measurement_time(Duration::from_secs(10));

    // IRI creation
    group.bench_function("named_node_iri", |b: &mut criterion::Bencher| {
        b.iter(|| {
            black_box(
                NamedNode::new(black_box("http://example.org/subject/resource"))
                    .expect("valid IRI"),
            )
        });
    });

    // Long IRI creation
    group.bench_function("named_node_long_iri", |b: &mut criterion::Bencher| {
        let iri = "http://www.w3.org/2002/07/owl#DatatypeProperty/long/namespace/path/resource";
        b.iter(|| black_box(NamedNode::new(black_box(iri)).expect("valid IRI")));
    });

    // Plain literal creation
    group.bench_function("literal_plain", |b: &mut criterion::Bencher| {
        b.iter(|| black_box(Literal::new(black_box("Hello, world!"))));
    });

    // Typed literal creation
    group.bench_function("literal_typed", |b: &mut criterion::Bencher| {
        let xsd_int =
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("valid XSD IRI");
        b.iter(|| black_box(Literal::new_typed(black_box("42"), xsd_int.clone())));
    });

    // Language-tagged literal creation
    group.bench_function("literal_lang_tagged", |b: &mut criterion::Bencher| {
        b.iter(|| black_box(Literal::new_lang(black_box("Hello"), black_box("en"))));
    });

    // Blank node creation
    group.bench_function("blank_node", |b: &mut criterion::Bencher| {
        b.iter(|| black_box(BlankNode::new(black_box("b1")).expect("valid blank node")));
    });

    group.finish();
}

// --- Triple construction benchmarks ---

fn bench_triple_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf/triple_construction");
    group.measurement_time(Duration::from_secs(10));

    let s = NamedNode::new("http://example.org/subject").expect("valid IRI");
    let p = NamedNode::new("http://example.org/predicate").expect("valid IRI");
    let o_lit = Literal::new("object value");
    let o_node = NamedNode::new("http://example.org/object").expect("valid IRI");

    // Triple with literal object
    group.bench_function(
        "triple_subject_pred_literal",
        |b: &mut criterion::Bencher| {
            b.iter(|| {
                black_box(Triple::new(
                    black_box(s.clone()),
                    black_box(p.clone()),
                    black_box(o_lit.clone()),
                ))
            });
        },
    );

    // Triple with IRI object
    group.bench_function("triple_subject_pred_iri", |b: &mut criterion::Bencher| {
        b.iter(|| {
            black_box(Triple::new(
                black_box(s.clone()),
                black_box(p.clone()),
                black_box(o_node.clone()),
            ))
        });
    });

    // Quad (triple + graph name)
    group.bench_function("quad_default_graph", |b: &mut criterion::Bencher| {
        b.iter(|| {
            black_box(Quad::new_default_graph(
                black_box(s.clone()),
                black_box(p.clone()),
                black_box(o_lit.clone()),
            ))
        });
    });

    group.finish();
}

// --- Graph insert benchmarks ---

fn bench_graph_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf/graph_insert");
    group.measurement_time(Duration::from_secs(12));

    for insert_count in [100usize, 1_000, 10_000] {
        // Pre-build triples outside the timed loop
        let triples: Vec<Triple> = (0..insert_count)
            .map(|i| {
                Triple::new(
                    NamedNode::new(format!("http://example.org/s/{i}")).expect("valid IRI"),
                    NamedNode::new(format!("http://example.org/p/{}", i % 20)).expect("valid IRI"),
                    Literal::new(format!("value {i}")),
                )
            })
            .collect();

        group.throughput(Throughput::Elements(insert_count as u64));
        group.bench_with_input(
            BenchmarkId::new("indexed_graph", insert_count),
            &insert_count,
            |b, &n| {
                b.iter(|| {
                    let graph = IndexedGraph::new();
                    for triple in &triples[..n] {
                        graph.insert(black_box(triple));
                    }
                    graph.len()
                });
            },
        );
    }

    // Also benchmark ConcreteStore (SPARQL quad store)
    for insert_count in [100usize, 1_000] {
        let quads: Vec<Quad> = (0..insert_count)
            .map(|i| {
                Quad::new_default_graph(
                    NamedNode::new(format!("http://example.org/s/{i}")).expect("valid IRI"),
                    NamedNode::new(format!("http://example.org/p/{}", i % 20)).expect("valid IRI"),
                    Literal::new(format!("value {i}")),
                )
            })
            .collect();

        group.throughput(Throughput::Elements(insert_count as u64));
        group.bench_with_input(
            BenchmarkId::new("concrete_store", insert_count),
            &insert_count,
            |b, &n| {
                b.iter(|| {
                    let store = ConcreteStore::new().expect("create store");
                    let mut count = 0usize;
                    for quad in &quads[..n] {
                        if store
                            .insert_quad(black_box(quad.clone()))
                            .expect("insert quad")
                        {
                            count += 1;
                        }
                    }
                    count
                });
            },
        );
    }

    group.finish();
}

// --- Triple pattern matching benchmarks ---

fn bench_pattern_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf/pattern_match");
    group.measurement_time(Duration::from_secs(12));

    // Setup a graph with 10k triples (100 subjects x 10 predicates x 10 objects)
    let graph = IndexedGraph::new();
    for s in 0..100usize {
        for p in 0..10usize {
            for o in 0..10usize {
                let triple = Triple::new(
                    NamedNode::new(format!("http://example.org/s/{s}")).expect("valid IRI"),
                    NamedNode::new(format!("http://example.org/p/{p}")).expect("valid IRI"),
                    Literal::new(format!("object {o}")),
                );
                graph.insert(&triple);
            }
        }
    }

    let subject_50 =
        Subject::NamedNode(NamedNode::new("http://example.org/s/50").expect("valid IRI"));
    let predicate_5 =
        Predicate::NamedNode(NamedNode::new("http://example.org/p/5").expect("valid IRI"));
    let object_3 = Object::Literal(Literal::new("object 3"));

    // Query by subject (returns 100 triples)
    group.bench_function("by_subject", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let results = graph.query(Some(black_box(&subject_50)), None, None);
            black_box(results.len())
        });
    });

    // Query by predicate (returns 1000 triples)
    group.bench_function("by_predicate", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let results = graph.query(None, Some(black_box(&predicate_5)), None);
            black_box(results.len())
        });
    });

    // Query by object (returns 100 triples)
    group.bench_function("by_object", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let results = graph.query(None, None, Some(black_box(&object_3)));
            black_box(results.len())
        });
    });

    // Query by subject + predicate (returns 10 triples)
    group.bench_function("by_subject_and_predicate", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let results = graph.query(
                Some(black_box(&subject_50)),
                Some(black_box(&predicate_5)),
                None,
            );
            black_box(results.len())
        });
    });

    // Full triple lookup (returns 1 triple)
    group.bench_function("full_triple_lookup", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let results = graph.query(
                Some(black_box(&subject_50)),
                Some(black_box(&predicate_5)),
                Some(black_box(&object_3)),
            );
            black_box(results.len())
        });
    });

    group.finish();
}

// --- IRI parsing benchmarks ---

fn bench_iri_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf/iri_parsing");
    group.measurement_time(Duration::from_secs(10));

    let iris: &[(&str, &str)] = &[
        ("short", "http://example.org/x"),
        ("medium", "http://www.w3.org/2002/07/owl#Class"),
        (
            "long",
            "http://www.w3.org/ns/hydra/core#ApiDocumentation/endpoint/resource/v2",
        ),
        (
            "rdf_type",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        ),
        (
            "xsd_datetime",
            "http://www.w3.org/2001/XMLSchema#dateTimeStamp",
        ),
    ];

    for (label, iri) in iris {
        group.bench_with_input(BenchmarkId::from_parameter(label), iri, |b, iri| {
            b.iter(|| black_box(NamedNode::new(black_box(*iri)).expect("valid IRI")));
        });
    }

    // Repeated IRI parsing (same string, hot cache scenario)
    group.bench_function("repeated_same_iri", |b: &mut criterion::Bencher| {
        let iri = "http://xmlns.com/foaf/0.1/Person";
        b.iter(|| {
            for _ in 0..100 {
                black_box(NamedNode::new(black_box(iri)).expect("valid IRI"));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_term_creation,
    bench_triple_construction,
    bench_graph_insert,
    bench_pattern_match,
    bench_iri_parsing,
);
criterion_main!(benches);
