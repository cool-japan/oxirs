//! Performance benchmarks for oxirs-ttl optimizations
//!
//! This benchmark suite measures the impact of our performance optimizations:
//! - String interning for IRI deduplication
//! - Buffer management for memory pooling
//! - SIMD-accelerated fast scanning
//!
//! Run with: cargo bench --bench performance_benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_ttl::formats::turtle::TurtleParser;
use oxirs_ttl::toolkit::{BufferManager, FastScanner, Parser, StringInterner};
use std::hint::black_box;

// Generate realistic RDF data for benchmarking
fn generate_turtle_data(num_triples: usize) -> String {
    let mut turtle = String::from("@prefix ex: <http://example.org/> .\n");
    turtle.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
    turtle.push_str("@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n\n");

    for i in 0..num_triples {
        turtle.push_str(&format!("ex:person{} rdf:type ex:Person .\n", i));
        turtle.push_str(&format!("ex:person{} foaf:name \"Person {}\" .\n", i, i));
        turtle.push_str(&format!("ex:person{} foaf:age {} .\n", i, 20 + (i % 50)));
    }

    turtle
}

// Benchmark string interning performance
fn bench_string_interning(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_interning");

    // Common RDF predicates that appear frequently
    let common_iris = vec![
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2000/01/rdf-schema#comment",
        "http://xmlns.com/foaf/0.1/name",
        "http://xmlns.com/foaf/0.1/age",
        "http://example.org/knows",
    ];

    // Benchmark: Intern common IRIs repeatedly (simulates parsing)
    group.bench_function("intern_common_iris", |b| {
        b.iter(|| {
            let mut interner = StringInterner::new();
            for _ in 0..100 {
                for iri in &common_iris {
                    black_box(interner.intern(iri));
                }
            }
        });
    });

    // Benchmark: With pre-populated namespaces
    group.bench_function("intern_with_prepopulated", |b| {
        b.iter(|| {
            let mut interner = StringInterner::with_common_namespaces();
            for _ in 0..100 {
                for iri in &common_iris {
                    black_box(interner.intern(iri));
                }
            }
        });
    });

    group.finish();
}

// Benchmark buffer management performance
fn bench_buffer_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_management");

    // Benchmark: Buffer pool vs fresh allocations
    group.bench_function("buffer_pool_reuse", |b| {
        b.iter(|| {
            let mut manager = BufferManager::new();
            for _ in 0..1000 {
                let buffer = manager.acquire_string_buffer();
                black_box(&buffer);
                manager.release_string_buffer(buffer);
            }
        });
    });

    group.bench_function("fresh_allocations", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let buffer = String::with_capacity(256);
                black_box(&buffer);
                // Buffer is dropped
            }
        });
    });

    // Benchmark: Blank node ID generation
    group.bench_function("generate_blank_node_ids", |b| {
        b.iter(|| {
            let mut manager = BufferManager::new();
            for i in 0..1000 {
                let id = manager.generate_blank_node_id(i);
                black_box(&id);
            }
        });
    });

    group.finish();
}

// Benchmark SIMD-accelerated fast scanner
fn bench_fast_scanner(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_scanner");

    // Large input for meaningful SIMD performance
    let large_input = "   ".repeat(1000) + "<http://example.org/resource> ";
    let input_with_comments = "  # comment 1\n  # comment 2\n  <http://example.org/> ";

    // Benchmark: Skip whitespace
    group.throughput(Throughput::Bytes(large_input.len() as u64));
    group.bench_function("skip_whitespace", |b| {
        let scanner = FastScanner::new(large_input.as_bytes());
        b.iter(|| {
            let pos = scanner.skip_whitespace(0);
            black_box(pos);
        });
    });

    // Benchmark: Skip whitespace and comments
    group.bench_function("skip_whitespace_and_comments", |b| {
        let scanner = FastScanner::new(input_with_comments.as_bytes());
        b.iter(|| {
            let pos = scanner.skip_whitespace_and_comments(0);
            black_box(pos);
        });
    });

    // Benchmark: Find byte (SIMD search)
    group.bench_function("find_byte_simd", |b| {
        let scanner = FastScanner::new(large_input.as_bytes());
        b.iter(|| {
            let pos = scanner.find_byte(b'<', 0);
            black_box(pos);
        });
    });

    // Benchmark: Scan IRI reference
    let iri_input = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>";
    group.bench_function("scan_iri_ref", |b| {
        let scanner = FastScanner::new(iri_input.as_bytes());
        b.iter(|| {
            let end = scanner.scan_iri_ref(0);
            black_box(end);
        });
    });

    // Benchmark: Scan string literal
    let string_input = r#""This is a test string with some content""#;
    group.bench_function("scan_string_literal", |b| {
        let scanner = FastScanner::new(string_input.as_bytes());
        b.iter(|| {
            let end = scanner.scan_string_literal(0, b'"');
            black_box(end);
        });
    });

    // Benchmark: Scan prefixed name
    let prefixed_input = "foaf:name ";
    group.bench_function("scan_prefixed_name", |b| {
        let scanner = FastScanner::new(prefixed_input.as_bytes());
        b.iter(|| {
            let result = scanner.scan_prefixed_name(0);
            black_box(result);
        });
    });

    group.finish();
}

// Benchmark full parsing with different dataset sizes
fn bench_full_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_parsing");

    for num_triples in [10, 100, 1000].iter() {
        let turtle_data = generate_turtle_data(*num_triples);
        let data_size = turtle_data.len();

        group.throughput(Throughput::Bytes(data_size as u64));
        group.bench_with_input(
            BenchmarkId::new("parse_turtle", num_triples),
            num_triples,
            |b, _| {
                let parser = TurtleParser::new();
                b.iter(|| {
                    let result = parser.parse(turtle_data.as_bytes());
                    let _ = black_box(result);
                });
            },
        );
    }

    group.finish();
}

// Benchmark parsing with realistic RDF patterns
fn bench_realistic_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_parsing");

    // Real-world RDF patterns
    let foaf_data = r#"
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix ex: <http://example.org/> .

ex:alice foaf:name "Alice" ;
         foaf:age 30 ;
         foaf:knows ex:bob .

ex:bob foaf:name "Bob" ;
       foaf:age 25 ;
       foaf:knows ex:charlie .

ex:charlie foaf:name "Charlie" ;
           foaf:age 35 ;
           foaf:knows ex:alice .
"#;

    group.bench_function("parse_foaf_network", |b| {
        let parser = TurtleParser::new();
        b.iter(|| {
            let result = parser.parse(foaf_data.as_bytes());
            let _ = black_box(result);
        });
    });

    // RDF Schema data
    let rdfs_data = r#"
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

ex:Person rdf:type rdfs:Class ;
          rdfs:label "Person" ;
          rdfs:comment "A human being" .

ex:name rdf:type rdf:Property ;
        rdfs:domain ex:Person ;
        rdfs:range rdfs:Literal .

ex:age rdf:type rdf:Property ;
       rdfs:domain ex:Person ;
       rdfs:range rdfs:Literal .
"#;

    group.bench_function("parse_rdfs_schema", |b| {
        let parser = TurtleParser::new();
        b.iter(|| {
            let result = parser.parse(rdfs_data.as_bytes());
            let _ = black_box(result);
        });
    });

    group.finish();
}

// Benchmark memory efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Measure string interning efficiency with repeated predicates
    let repeated_predicates = r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"#
    .to_string()
        + &(0..500)
            .map(|i| format!("ex:entity{} rdf:type ex:Thing .\n", i))
            .collect::<String>();

    group.bench_function("parse_with_repeated_predicates", |b| {
        let parser = TurtleParser::new();
        b.iter(|| {
            let result = parser.parse(repeated_predicates.as_bytes());
            let _ = black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_string_interning,
    bench_buffer_management,
    bench_fast_scanner,
    bench_full_parsing,
    bench_realistic_parsing,
    bench_memory_efficiency
);

criterion_main!(benches);
