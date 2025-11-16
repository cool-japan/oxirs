//! Performance benchmarks for CLI operations
//!
//! Validates performance characteristics of core RDF operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_core::format::{RdfFormat, RdfParser, RdfSerializer};
use oxirs_core::model::{GraphName, NamedNode, Quad, Subject, Term};
use std::io::Cursor;

/// Generate sample RDF data for benchmarking
fn generate_sample_quads(count: usize) -> Vec<Quad> {
    (0..count)
        .map(|i| {
            let subject = NamedNode::new(format!("http://example.org/subject{}", i)).unwrap();
            let predicate = NamedNode::new("http://example.org/predicate").unwrap();
            let object = Term::Literal(oxirs_core::model::Literal::new(format!("Value {}", i)));

            Quad::new(
                Subject::NamedNode(subject),
                predicate,
                object,
                GraphName::DefaultGraph,
            )
        })
        .collect()
}

/// Benchmark RDF serialization performance
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    for size in [100, 1000, 10000].iter() {
        let quads = generate_sample_quads(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // Benchmark Turtle serialization
        group.bench_with_input(BenchmarkId::new("turtle", size), &quads, |b, quads| {
            b.iter(|| {
                let buffer = Vec::new();
                let mut serializer = RdfSerializer::new(RdfFormat::Turtle).for_writer(buffer);

                for quad in quads {
                    serializer.serialize_quad(quad.as_ref()).unwrap();
                }
                let buffer = serializer.finish().unwrap();
                black_box(buffer);
            });
        });

        // Benchmark N-Triples serialization
        group.bench_with_input(BenchmarkId::new("ntriples", size), &quads, |b, quads| {
            b.iter(|| {
                let buffer = Vec::new();
                let mut serializer = RdfSerializer::new(RdfFormat::NTriples).for_writer(buffer);

                for quad in quads {
                    serializer.serialize_quad(quad.as_ref()).unwrap();
                }
                let buffer = serializer.finish().unwrap();
                black_box(buffer);
            });
        });

        // Benchmark N-Quads serialization
        group.bench_with_input(BenchmarkId::new("nquads", size), &quads, |b, quads| {
            b.iter(|| {
                let buffer = Vec::new();
                let mut serializer = RdfSerializer::new(RdfFormat::NQuads).for_writer(buffer);

                for quad in quads {
                    serializer.serialize_quad(quad.as_ref()).unwrap();
                }
                let buffer = serializer.finish().unwrap();
                black_box(buffer);
            });
        });
    }

    group.finish();
}

/// Benchmark RDF parsing performance
fn bench_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing");

    for size in [100, 1000, 10000].iter() {
        // Generate sample N-Triples data
        let sample_data = (0..*size)
            .map(|i| {
                format!(
                    "<http://example.org/subject{}> <http://example.org/predicate> \"Value {}\" .\n",
                    i, i
                )
            })
            .collect::<String>();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("ntriples", size),
            &sample_data,
            |b, data| {
                b.iter(|| {
                    let cursor = Cursor::new(data.as_bytes().to_vec());
                    let parser = RdfParser::new(RdfFormat::NTriples);
                    let quads: Vec<_> = parser
                        .for_reader(cursor)
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    black_box(quads);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Turtle parsing performance
fn bench_turtle_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("turtle_parsing");

    for size in [100, 1000, 10000].iter() {
        // Generate sample Turtle data
        let mut turtle_data = String::from("@prefix ex: <http://example.org/> .\n\n");
        for i in 0..*size {
            turtle_data.push_str(&format!(
                "ex:subject{} ex:predicate{} \"Value {}\" .\n",
                i,
                i % 10,
                i
            ));
        }

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("turtle", size), &turtle_data, |b, data| {
            b.iter(|| {
                let cursor = Cursor::new(data.as_bytes().to_vec());
                let parser = RdfParser::new(RdfFormat::Turtle);
                let quads: Vec<_> = parser
                    .for_reader(cursor)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                black_box(quads);
            });
        });
    }

    group.finish();
}

/// Benchmark RDF/XML parsing performance
fn bench_rdfxml_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdfxml_parsing");

    for size in [100, 1000, 10000].iter() {
        // Generate sample RDF/XML data
        let mut rdfxml_data = String::from(
            r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:ex="http://example.org/">
"#,
        );

        for i in 0..*size {
            rdfxml_data.push_str(&format!(
                r#"    <rdf:Description rdf:about="http://example.org/subject{}">
        <ex:predicate{}>Value {}</ex:predicate{}>
    </rdf:Description>
"#,
                i,
                i % 10,
                i,
                i % 10
            ));
        }

        rdfxml_data.push_str("</rdf:RDF>");

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("rdfxml", size), &rdfxml_data, |b, data| {
            b.iter(|| {
                let cursor = Cursor::new(data.as_bytes().to_vec());
                let parser = RdfParser::new(RdfFormat::RdfXml);
                let quads: Vec<_> = parser
                    .for_reader(cursor)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                black_box(quads);
            });
        });
    }

    group.finish();
}

/// Benchmark JSON-LD parsing performance
fn bench_jsonld_parsing(c: &mut Criterion) {
    use oxirs_core::format::JsonLdProfileSet;

    let mut group = c.benchmark_group("jsonld_parsing");

    for size in [100, 1000, 10000].iter() {
        // Generate sample JSON-LD data
        let mut jsonld_objects = Vec::new();
        for i in 0..*size {
            jsonld_objects.push(format!(
                r#"{{
    "@id": "http://example.org/subject{}",
    "http://example.org/predicate{}": "Value {}"
}}"#,
                i,
                i % 10,
                i
            ));
        }

        let jsonld_data = format!("[{}]", jsonld_objects.join(",\n"));

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("jsonld", size), &jsonld_data, |b, data| {
            b.iter(|| {
                let cursor = Cursor::new(data.as_bytes().to_vec());
                let parser = RdfParser::new(RdfFormat::JsonLd {
                    profile: JsonLdProfileSet::empty(),
                });
                let quads: Vec<_> = parser
                    .for_reader(cursor)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                black_box(quads);
            });
        });
    }

    group.finish();
}

/// Benchmark format conversion (parse + serialize)
fn bench_format_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_conversion");

    for size in [100, 1000, 10000].iter() {
        // Generate sample N-Triples data
        let sample_data = (0..*size)
            .map(|i| {
                format!(
                    "<http://example.org/subject{}> <http://example.org/predicate> \"Value {}\" .\n",
                    i, i
                )
            })
            .collect::<String>();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("ntriples_to_turtle", size),
            &sample_data,
            |b, data| {
                b.iter(|| {
                    // Parse N-Triples
                    let cursor = Cursor::new(data.as_bytes().to_vec());
                    let parser = RdfParser::new(RdfFormat::NTriples);
                    let quads: Vec<_> = parser
                        .for_reader(cursor)
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();

                    // Serialize to Turtle
                    let buffer = Vec::new();
                    let mut serializer = RdfSerializer::new(RdfFormat::Turtle).for_writer(buffer);

                    for quad in &quads {
                        serializer.serialize_quad(quad.as_ref()).unwrap();
                    }
                    let buffer = serializer.finish().unwrap();
                    black_box(buffer);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency of streaming operations
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test with large dataset to verify streaming behavior
    let size = 50000;
    let sample_data = (0..size)
        .map(|i| {
            format!(
                "<http://example.org/subject{}> <http://example.org/predicate> \"Value {}\" .\n",
                i, i
            )
        })
        .collect::<String>();

    group.throughput(Throughput::Elements(size as u64));
    group.sample_size(10); // Reduce sample size for large benchmarks

    group.bench_function("streaming_parse_serialize", |b| {
        b.iter(|| {
            let cursor = Cursor::new(sample_data.as_bytes().to_vec());
            let parser = RdfParser::new(RdfFormat::NTriples);

            let buffer = Vec::new();
            let mut serializer = RdfSerializer::new(RdfFormat::Turtle).for_writer(buffer);

            // Stream through data without collecting
            for quad_result in parser.for_reader(cursor) {
                let quad = quad_result.unwrap();
                serializer.serialize_quad(quad.as_ref()).unwrap();
            }
            let buffer = serializer.finish().unwrap();
            black_box(buffer);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_serialization,
    bench_parsing,
    bench_turtle_parsing,
    bench_rdfxml_parsing,
    bench_jsonld_parsing,
    bench_format_conversion,
    bench_memory_efficiency
);
criterion_main!(benches);
