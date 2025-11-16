//! Benchmark suite for RDF parsers in oxirs-ttl
//!
//! Run with: cargo bench -p oxirs-ttl

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_ttl::nquads::NQuadsParser;
use oxirs_ttl::ntriples::NTriplesParser;
use oxirs_ttl::trig::TriGParser;
use oxirs_ttl::turtle::TurtleParser;
use oxirs_ttl::Parser;
use std::hint::black_box;
use std::io::Cursor;

/// Generate N-Triples test data
fn generate_ntriples(count: usize) -> String {
    let mut data = String::new();
    for i in 0..count {
        data.push_str(&format!(
            "<http://example.org/subject{}> <http://example.org/predicate> \"object{}\" .\n",
            i, i
        ));
    }
    data
}

/// Generate N-Quads test data
fn generate_nquads(count: usize) -> String {
    let mut data = String::new();
    for i in 0..count {
        let graph = if i % 10 == 0 {
            format!("<http://example.org/graph{}>", i / 10)
        } else {
            String::new()
        };
        data.push_str(&format!(
            "<http://example.org/subject{}> <http://example.org/predicate> \"object{}\" {} .\n",
            i, i, graph
        ));
    }
    data
}

/// Generate Turtle test data
fn generate_turtle(count: usize) -> String {
    let mut data = String::from("@prefix ex: <http://example.org/> .\n\n");
    for i in 0..count {
        data.push_str(&format!("ex:subject{} ex:predicate \"object{}\" .\n", i, i));
    }
    data
}

/// Generate TriG test data
fn generate_trig(count: usize) -> String {
    let mut data = String::from("@prefix ex: <http://example.org/> .\n\n");
    data.push_str("<http://example.org/graph1> {\n");
    for i in 0..count {
        data.push_str(&format!(
            "    ex:subject{} ex:predicate \"object{}\" .\n",
            i, i
        ));
    }
    data.push_str("}\n");
    data
}

/// Benchmark N-Triples parser with various sizes
fn bench_ntriples_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntriples_parser");

    for size in [100, 1_000, 10_000].iter() {
        let data = generate_ntriples(*size);
        let size_bytes = data.len();

        group.throughput(Throughput::Bytes(size_bytes as u64));
        let data_bytes = data.into_bytes();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, move |b, _| {
            b.iter(|| {
                let parser = NTriplesParser::new();
                let bytes_for_bench = data_bytes.clone();
                let result: Result<Vec<_>, _> =
                    parser.for_reader(Cursor::new(bytes_for_bench)).collect();
                assert!(result.is_ok());
                black_box(result.unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark N-Quads parser with various sizes
fn bench_nquads_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("nquads_parser");

    for size in [100, 1_000, 10_000].iter() {
        let data = generate_nquads(*size);
        let size_bytes = data.len();

        group.throughput(Throughput::Bytes(size_bytes as u64));
        let data_bytes = data.into_bytes();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, move |b, _| {
            b.iter(|| {
                let parser = NQuadsParser::new();
                let bytes_for_bench = data_bytes.clone();
                let result: Result<Vec<_>, _> =
                    parser.for_reader(Cursor::new(bytes_for_bench)).collect();
                assert!(result.is_ok());
                black_box(result.unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark Turtle parser with various sizes
fn bench_turtle_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("turtle_parser");

    for size in [100, 1_000, 10_000].iter() {
        let data = generate_turtle(*size);
        let size_bytes = data.len();

        group.throughput(Throughput::Bytes(size_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let data_ref = &data;
            b.iter(|| {
                let parser = TurtleParser::new();
                let result = parser.parse_document(data_ref);
                assert!(result.is_ok());
                black_box(result.unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark TriG parser with various sizes
fn bench_trig_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("trig_parser");

    for size in [100, 1_000, 10_000].iter() {
        let data = generate_trig(*size);
        let size_bytes = data.len();

        group.throughput(Throughput::Bytes(size_bytes as u64));
        let data_bytes = data.into_bytes();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, move |b, _| {
            b.iter(|| {
                let parser = TriGParser::new();
                let bytes_for_bench = data_bytes.clone();
                let result: Result<Vec<_>, _> =
                    parser.for_reader(Cursor::new(bytes_for_bench)).collect();
                // TriG parser may not be fully functional yet
                black_box(result.ok())
            });
        });
    }

    group.finish();
}

/// Benchmark streaming parser performance
fn bench_streaming_parser(c: &mut Criterion) {
    use oxirs_ttl::streaming::{StreamingConfig, StreamingParser};

    let mut group = c.benchmark_group("streaming_parser");

    for size in [1_000, 10_000, 100_000].iter() {
        let data = generate_turtle(*size);
        let size_bytes = data.len();

        group.throughput(Throughput::Bytes(size_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let data_ref = &data;
            b.iter(|| {
                let config = StreamingConfig::default().with_batch_size(1000);
                let mut parser =
                    StreamingParser::with_config(Cursor::new(data_ref.as_str()), config);

                let mut total = 0;
                while let Some(batch) = parser.next_batch().ok().flatten() {
                    total += batch.len();
                }
                black_box(total)
            });
        });
    }

    group.finish();
}

/// Benchmark different batch sizes for streaming
fn bench_streaming_batch_sizes(c: &mut Criterion) {
    use oxirs_ttl::streaming::{StreamingConfig, StreamingParser};

    let data = generate_turtle(10_000);
    let mut group = c.benchmark_group("streaming_batch_sizes");

    let data_ref = &data;
    for batch_size in [100, 1_000, 5_000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = StreamingConfig::default().with_batch_size(batch_size);
                    let mut parser =
                        StreamingParser::with_config(Cursor::new(data_ref.as_str()), config);

                    let mut total = 0;
                    while let Some(batch) = parser.next_batch().ok().flatten() {
                        total += batch.len();
                    }
                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parser with complex Turtle features
fn bench_complex_turtle(c: &mut Criterion) {
    let complex_data = r#"
        @prefix ex: <http://example.org/> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:alice a foaf:Person ;
                 foaf:name "Alice" ;
                 foaf:age "30"^^xsd:integer ;
                 foaf:knows ex:bob , ex:charlie ;
                 foaf:homepage <http://alice.example.org/> .

        ex:bob a foaf:Person ;
               foaf:name "Bob" ;
               foaf:age "25"^^xsd:integer ;
               foaf:knows ex:alice .

        ex:charlie a foaf:Person ;
                   foaf:name "Charlie" ;
                   foaf:mbox <mailto:charlie@example.org> .
    "#;

    c.bench_function("complex_turtle", |b| {
        b.iter(|| {
            let parser = TurtleParser::new();
            let result = parser.parse_document(complex_data);
            assert!(result.is_ok());
            black_box(result.unwrap())
        });
    });
}

criterion_group!(
    benches,
    bench_ntriples_parser,
    bench_nquads_parser,
    bench_turtle_parser,
    bench_trig_parser,
    bench_streaming_parser,
    bench_streaming_batch_sizes,
    bench_complex_turtle
);

criterion_main!(benches);
