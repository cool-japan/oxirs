//! Performance benchmarks for RDF 1.2 features
//!
//! This benchmark suite measures the performance of:
//! - RDF-star (quoted triple) parsing and serialization
//! - Directional language tag processing
//! - Mixed RDF 1.1 and RDF 1.2 workloads
//! - Nested quoted triple structures

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_ttl::formats::turtle::{TurtleParser, TurtleSerializer};
use oxirs_ttl::toolkit::{Parser, Serializer};
use std::hint::black_box;

// ============================================================================
// Helper Functions
// ============================================================================

fn generate_quoted_triple_document(count: usize) -> String {
    let mut doc = String::from("@prefix ex: <http://example.org/> .\n\n");
    for i in 0..count {
        doc.push_str(&format!(
            "<< ex:subject{} ex:predicate{} ex:object{} >> ex:confidence \"high\" .\n",
            i, i, i
        ));
    }
    doc
}

fn generate_directional_tags_document(count: usize) -> String {
    let mut doc = String::from("@prefix ex: <http://example.org/> .\n\n");
    for i in 0..count {
        if i % 2 == 0 {
            doc.push_str(&format!(
                "ex:subject{} ex:nameEn \"Name {}\"@en--ltr .\n",
                i, i
            ));
        } else {
            doc.push_str(&format!(
                "ex:subject{} ex:nameAr \"الاسم {}\"@ar--rtl .\n",
                i, i
            ));
        }
    }
    doc
}

fn generate_mixed_rdf12_document(count: usize) -> String {
    let mut doc = String::from("@prefix ex: <http://example.org/> .\n\n");
    for i in 0..count {
        match i % 4 {
            0 => {
                // Quoted triple
                doc.push_str(&format!(
                    "<< ex:s{} ex:p{} ex:o{} >> ex:confidence \"high\" .\n",
                    i, i, i
                ));
            }
            1 => {
                // Directional language tag
                doc.push_str(&format!(
                    "ex:subject{} ex:text \"Text {}\"@en--ltr .\n",
                    i, i
                ));
            }
            2 => {
                // Standard triple
                doc.push_str(&format!(
                    "ex:subject{} ex:predicate{} ex:object{} .\n",
                    i, i, i
                ));
            }
            _ => {
                // Typed literal
                doc.push_str(&format!(
                    "ex:subject{} ex:value {}^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
                    i, i
                ));
            }
        }
    }
    doc
}

fn generate_nested_quoted_triples(depth: usize, count: usize) -> String {
    let mut doc = String::from("@prefix ex: <http://example.org/> .\n\n");

    for i in 0..count {
        let mut nested = format!("<< ex:a{} ex:b{} ex:c{} >>", i, i, i);
        for j in 0..depth {
            nested = format!("<< {} ex:p{} ex:o{} >>", nested, j, j);
        }
        doc.push_str(&format!("{} ex:final ex:value{} .\n", nested, i));
    }

    doc
}

// ============================================================================
// Quoted Triple Parsing Benchmarks
// ============================================================================

fn bench_quoted_triple_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("quoted_triple_parsing");

    for size in [10, 100, 1000].iter() {
        let doc = generate_quoted_triple_document(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &doc, |b, doc| {
            let parser = TurtleParser::new();
            b.iter(|| {
                let triples = parser.parse(black_box(doc.as_bytes())).unwrap();
                black_box(triples);
            });
        });
    }

    group.finish();
}

fn bench_nested_quoted_triple_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_quoted_triple_parsing");

    for depth in [1, 3, 5].iter() {
        let doc = generate_nested_quoted_triples(*depth, 100);
        group.throughput(Throughput::Elements(100));

        group.bench_with_input(BenchmarkId::new("depth", depth), &doc, |b, doc| {
            let parser = TurtleParser::new();
            b.iter(|| {
                let triples = parser.parse(black_box(doc.as_bytes())).unwrap();
                black_box(triples);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Directional Language Tag Benchmarks
// ============================================================================

fn bench_directional_language_tag_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("directional_language_tag_parsing");

    for size in [10, 100, 1000].iter() {
        let doc = generate_directional_tags_document(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &doc, |b, doc| {
            let parser = TurtleParser::new();
            b.iter(|| {
                let triples = parser.parse(black_box(doc.as_bytes())).unwrap();
                black_box(triples);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Mixed RDF 1.2 Workload Benchmarks
// ============================================================================

fn bench_mixed_rdf12_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_rdf12_parsing");

    for size in [10, 100, 1000].iter() {
        let doc = generate_mixed_rdf12_document(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &doc, |b, doc| {
            let parser = TurtleParser::new();
            b.iter(|| {
                let triples = parser.parse(black_box(doc.as_bytes())).unwrap();
                black_box(triples);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Quoted Triple Serialization Benchmarks
// ============================================================================

fn bench_quoted_triple_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quoted_triple_serialization");

    for size in [10, 100, 1000].iter() {
        let doc = generate_quoted_triple_document(*size);
        let parser = TurtleParser::new();
        let triples = parser.parse(doc.as_bytes()).unwrap();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &triples, |b, triples| {
            let serializer = TurtleSerializer::new();
            b.iter(|| {
                let mut output = Vec::new();
                serializer
                    .serialize(black_box(triples), &mut output)
                    .unwrap();
                black_box(output);
            });
        });
    }

    group.finish();
}

fn bench_directional_tag_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("directional_tag_serialization");

    for size in [10, 100, 1000].iter() {
        let doc = generate_directional_tags_document(*size);
        let parser = TurtleParser::new();
        let triples = parser.parse(doc.as_bytes()).unwrap();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &triples, |b, triples| {
            let serializer = TurtleSerializer::new();
            b.iter(|| {
                let mut output = Vec::new();
                serializer
                    .serialize(black_box(triples), &mut output)
                    .unwrap();
                black_box(output);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Round-trip Benchmarks
// ============================================================================

fn bench_rdf12_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf12_roundtrip");

    for size in [10, 100, 1000].iter() {
        let doc = generate_mixed_rdf12_document(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &doc, |b, doc| {
            b.iter(|| {
                // Parse
                let parser = TurtleParser::new();
                let triples = parser.parse(black_box(doc.as_bytes())).unwrap();

                // Serialize
                let serializer = TurtleSerializer::new();
                let mut output = Vec::new();
                serializer.serialize(&triples, &mut output).unwrap();

                // Re-parse
                let triples2 = parser.parse(black_box(&output[..])).unwrap();
                black_box(triples2);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Memory Efficiency Benchmarks
// ============================================================================

fn bench_rdf12_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf12_memory_usage");
    group.sample_size(20); // Reduce sample size for memory tests

    // Large dataset to measure memory efficiency
    let doc = generate_mixed_rdf12_document(10_000);
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("large_mixed_rdf12", |b| {
        b.iter(|| {
            let parser = TurtleParser::new();
            let triples = parser.parse(black_box(doc.as_bytes())).unwrap();
            black_box(triples);
        });
    });

    group.finish();
}

// ============================================================================
// Comparative Benchmarks (RDF 1.1 vs RDF 1.2)
// ============================================================================

fn bench_rdf11_vs_rdf12_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf11_vs_rdf12_comparison");

    let size = 1000;

    // RDF 1.1 document (no RDF 1.2 features)
    let mut rdf11_doc = String::from("@prefix ex: <http://example.org/> .\n\n");
    for i in 0..size {
        rdf11_doc.push_str(&format!(
            "ex:subject{} ex:predicate{} ex:object{} .\n",
            i, i, i
        ));
    }

    // RDF 1.2 document with quoted triples
    let rdf12_doc = generate_quoted_triple_document(size);

    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("rdf11_standard", |b| {
        let parser = TurtleParser::new();
        b.iter(|| {
            let triples = parser.parse(black_box(rdf11_doc.as_bytes())).unwrap();
            black_box(triples);
        });
    });

    group.bench_function("rdf12_quoted_triples", |b| {
        let parser = TurtleParser::new();
        b.iter(|| {
            let triples = parser.parse(black_box(rdf12_doc.as_bytes())).unwrap();
            black_box(triples);
        });
    });

    group.finish();
}

// ============================================================================
// Real-world Scenario Benchmarks
// ============================================================================

fn bench_knowledge_graph_with_provenance(c: &mut Criterion) {
    // Simulate a knowledge graph with provenance using quoted triples
    let doc = r#"
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Facts with provenance metadata using RDF-star
<< ex:alice ex:knows ex:bob >> ex:since "2020-01-01"^^xsd:date .
<< ex:alice ex:knows ex:bob >> ex:confidence "0.95"^^xsd:decimal .
<< ex:alice ex:knows ex:bob >> ex:source ex:socialNetwork .

<< ex:bob ex:knows ex:charlie >> ex:since "2021-06-15"^^xsd:date .
<< ex:bob ex:knows ex:charlie >> ex:confidence "0.87"^^xsd:decimal .
<< ex:bob ex:knows ex:charlie >> ex:source ex:emailArchive .

<< ex:charlie ex:knows ex:alice >> ex:since "2019-03-20"^^xsd:date .
<< ex:charlie ex:knows ex:alice >> ex:confidence "0.99"^^xsd:decimal .
<< ex:charlie ex:knows ex:alice >> ex:source ex:meetingRecords .

# Multilingual labels with directional tags
ex:alice ex:label "Alice"@en--ltr .
ex:alice ex:label "أليس"@ar--rtl .
ex:bob ex:label "Bob"@en--ltr .
ex:bob ex:label "بوب"@ar--rtl .
ex:charlie ex:label "Charlie"@en--ltr .
ex:charlie ex:label "تشارلي"@ar--rtl .
"#;

    c.bench_function("knowledge_graph_with_provenance", |b| {
        let parser = TurtleParser::new();
        b.iter(|| {
            let triples = parser.parse(black_box(doc.as_bytes())).unwrap();
            black_box(triples);
        });
    });
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    rdf12_parsing_benches,
    bench_quoted_triple_parsing,
    bench_nested_quoted_triple_parsing,
    bench_directional_language_tag_parsing,
    bench_mixed_rdf12_parsing,
    bench_rdf11_vs_rdf12_parsing,
);

criterion_group!(
    rdf12_serialization_benches,
    bench_quoted_triple_serialization,
    bench_directional_tag_serialization,
);

criterion_group!(
    rdf12_integration_benches,
    bench_rdf12_roundtrip,
    bench_rdf12_memory_usage,
    bench_knowledge_graph_with_provenance,
);

criterion_main!(
    rdf12_parsing_benches,
    rdf12_serialization_benches,
    rdf12_integration_benches,
);
