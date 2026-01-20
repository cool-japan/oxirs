//! Benchmark suite for RDF serializers in oxirs-ttl
//!
//! Run with: cargo bench -p oxirs-ttl --bench serializer_benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_core::model::{Literal, NamedNode, Triple};
use oxirs_ttl::ntriples::NTriplesSerializer;
use oxirs_ttl::turtle::TurtleSerializer;
use oxirs_ttl::Serializer;

/// Generate test triples
fn generate_triples(count: usize) -> Vec<Triple> {
    (0..count)
        .map(|i| {
            Triple::new(
                NamedNode::new_unchecked(format!("http://example.org/subject{}", i)),
                NamedNode::new_unchecked("http://example.org/predicate"),
                Literal::new_simple_literal(format!("object{}", i)),
            )
        })
        .collect()
}

/// Benchmark N-Triples serializer with various sizes
fn bench_ntriples_serializer(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntriples_serializer");

    for size in [100, 1_000, 10_000].iter() {
        let triples = generate_triples(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut buffer = Vec::new();
                let serializer = NTriplesSerializer::new();
                serializer.serialize(&triples, &mut buffer).unwrap();
                buffer
            });
        });
    }

    group.finish();
}

/// Benchmark Turtle serializer with various sizes
fn bench_turtle_serializer(c: &mut Criterion) {
    let mut group = c.benchmark_group("turtle_serializer");

    for size in [100, 1_000, 10_000].iter() {
        let triples = generate_triples(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut buffer = Vec::new();
                let serializer = TurtleSerializer::new();
                serializer.serialize(&triples, &mut buffer).unwrap();
                buffer
            });
        });
    }

    group.finish();
}

/// Benchmark roundtrip (parse + serialize) performance
fn bench_roundtrip_ntriples(c: &mut Criterion) {
    use oxirs_ttl::ntriples::NTriplesParser;
    use oxirs_ttl::Parser;
    use std::io::Cursor;

    let mut group = c.benchmark_group("roundtrip_ntriples");

    for size in [100, 1_000, 10_000].iter() {
        let triples = generate_triples(*size);

        // First serialize to get input data
        let mut buffer = Vec::new();
        let serializer = NTriplesSerializer::new();
        serializer.serialize(&triples, &mut buffer).unwrap();
        let data = String::from_utf8(buffer).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        let data_bytes = data.into_bytes();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, move |b, _| {
            b.iter(|| {
                // Parse
                let parser = NTriplesParser::new();
                let bytes_for_bench = data_bytes.clone();
                let parsed: Vec<_> = parser
                    .for_reader(Cursor::new(bytes_for_bench))
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();

                // Serialize
                let mut out_buffer = Vec::new();
                let out_serializer = NTriplesSerializer::new();
                out_serializer.serialize(&parsed, &mut out_buffer).unwrap();

                out_buffer
            });
        });
    }

    group.finish();
}

/// Benchmark serialization with different literal types
fn bench_serializer_literal_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("serializer_literal_types");

    // Simple literals
    let simple_triples: Vec<_> = (0..1000)
        .map(|i| {
            Triple::new(
                NamedNode::new_unchecked("http://example.org/subject"),
                NamedNode::new_unchecked("http://example.org/predicate"),
                Literal::new_simple_literal(format!("Simple literal {}", i)),
            )
        })
        .collect();

    group.bench_function("simple_literals", |b| {
        b.iter(|| {
            let mut buffer = Vec::new();
            let serializer = NTriplesSerializer::new();
            serializer.serialize(&simple_triples, &mut buffer).unwrap();
            buffer
        });
    });

    // Language-tagged literals
    let lang_triples: Vec<_> = (0..1000)
        .map(|i| {
            Triple::new(
                NamedNode::new_unchecked("http://example.org/subject"),
                NamedNode::new_unchecked("http://example.org/predicate"),
                Literal::new_language_tagged_literal_unchecked(
                    format!("Language literal {}", i),
                    "en",
                ),
            )
        })
        .collect();

    group.bench_function("language_tagged", |b| {
        b.iter(|| {
            let mut buffer = Vec::new();
            let serializer = NTriplesSerializer::new();
            serializer.serialize(&lang_triples, &mut buffer).unwrap();
            buffer
        });
    });

    // Typed literals
    let typed_triples: Vec<_> = (0..1000)
        .map(|i| {
            Triple::new(
                NamedNode::new_unchecked("http://example.org/subject"),
                NamedNode::new_unchecked("http://example.org/predicate"),
                Literal::new_typed_literal(
                    i.to_string(),
                    NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
                ),
            )
        })
        .collect();

    group.bench_function("typed_literals", |b| {
        b.iter(|| {
            let mut buffer = Vec::new();
            let serializer = NTriplesSerializer::new();
            serializer.serialize(&typed_triples, &mut buffer).unwrap();
            buffer
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ntriples_serializer,
    bench_turtle_serializer,
    bench_roundtrip_ntriples,
    bench_serializer_literal_types
);

criterion_main!(benches);
