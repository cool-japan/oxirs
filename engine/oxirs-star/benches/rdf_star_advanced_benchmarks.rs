//! Comprehensive benchmarks for advanced RDF-star features
//!
//! This benchmark suite validates the performance of:
//! - HDT-star binary format (encoding, decoding, compression)
//! - Streaming query processor (window operations, CEP)
//! - Property graph bridge (RDF-star ↔ LPG conversion)
//!
//! Run with: `cargo bench --bench rdf_star_advanced_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_star::hdt_star::{HdtStarBuilder, HdtStarConfig, HdtStarReader};
use oxirs_star::property_graph_bridge::{
    ConversionConfig, LabeledPropertyGraph, LpgEdge, LpgNode, PropertyGraphBridge, PropertyValue,
};
use oxirs_star::streaming_query::{CepMatcher, CepPattern, WindowConfig, WindowedAggregator};
use oxirs_star::{StarStore, StarTerm, StarTriple};
use std::io::Cursor;

// ============================================================================
// HDT-STAR FORMAT BENCHMARKS
// ============================================================================

/// Benchmark HDT-star encoding with various dataset sizes
fn bench_hdt_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdt_star/encoding");

    for size in [10, 100, 1_000, 10_000].iter() {
        let store = create_test_store(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let config = HdtStarConfig {
                    enable_compression: false,
                    ..HdtStarConfig::default()
                };
                let mut builder = HdtStarBuilder::new(config);
                builder.add_store(&store).unwrap();

                let mut buffer = Vec::new();
                builder.write(&mut buffer).unwrap();
                black_box(buffer)
            });
        });
    }

    group.finish();
}

/// Benchmark HDT-star decoding
fn bench_hdt_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdt_star/decoding");

    for size in [10, 100, 1_000, 10_000].iter() {
        let store = create_test_store(*size);
        let config = HdtStarConfig {
            enable_compression: false,
            ..HdtStarConfig::default()
        };
        let mut builder = HdtStarBuilder::new(config);
        builder.add_store(&store).unwrap();

        let mut buffer = Vec::new();
        builder.write(&mut buffer).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let mut cursor = Cursor::new(&buffer);
                let reader = HdtStarReader::read(&mut cursor).unwrap();
                black_box(reader.len())
            });
        });
    }

    group.finish();
}

/// Benchmark HDT-star compression ratios
fn bench_hdt_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdt_star/compression");

    let store = create_test_store(1_000);

    for level in [1, 3, 6, 9].iter() {
        group.bench_with_input(BenchmarkId::new("zstd", level), level, |b, &level| {
            b.iter(|| {
                let config = HdtStarConfig {
                    enable_compression: true,
                    compression_level: level,
                    ..HdtStarConfig::default()
                };
                let mut builder = HdtStarBuilder::new(config);
                builder.add_store(&store).unwrap();

                let mut buffer = Vec::new();
                builder.write(&mut buffer).unwrap();
                black_box(buffer.len())
            });
        });
    }

    group.finish();
}

/// Benchmark HDT-star roundtrip (encode + decode)
fn bench_hdt_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdt_star/roundtrip");

    for size in [100, 1_000].iter() {
        let store = create_test_store(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                // Encode
                let config = HdtStarConfig::default();
                let mut builder = HdtStarBuilder::new(config);
                builder.add_store(&store).unwrap();

                let mut buffer = Vec::new();
                builder.write(&mut buffer).unwrap();

                // Decode
                let mut cursor = Cursor::new(&buffer);
                let reader = HdtStarReader::read(&mut cursor).unwrap();
                black_box(reader.len())
            });
        });
    }

    group.finish();
}

/// Benchmark HDT-star with quoted triples
fn bench_hdt_quoted_triples(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdt_star/quoted_triples");

    for quoted_ratio in [0, 25, 50, 100].iter() {
        let store = create_quoted_triple_store(1_000, *quoted_ratio);

        group.bench_with_input(
            BenchmarkId::new("encoding", format!("{}%_quoted", quoted_ratio)),
            quoted_ratio,
            |b, &_ratio| {
                b.iter(|| {
                    let config = HdtStarConfig::default();
                    let mut builder = HdtStarBuilder::new(config);
                    builder.add_store(&store).unwrap();

                    let mut buffer = Vec::new();
                    builder.write(&mut buffer).unwrap();
                    black_box(buffer.len())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// STREAMING QUERY BENCHMARKS
// ============================================================================

/// Benchmark windowed aggregation
fn bench_windowed_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming/windowed_aggregation");

    for window_size in [10, 100, 1_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(window_size),
            window_size,
            |b, &size| {
                b.iter(|| {
                    let mut agg = WindowedAggregator::new(WindowConfig::count(size));

                    for i in 0..(size * 2) {
                        agg.add(i as f64);
                    }

                    black_box((agg.count(), agg.sum(), agg.avg()))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CEP pattern matching
fn bench_cep_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming/cep_matching");

    for event_count in [100, 1_000, 10_000].iter() {
        let triples = create_event_stream(*event_count);

        group.throughput(Throughput::Elements(*event_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(event_count),
            event_count,
            |b, &_count| {
                b.iter(|| {
                    let mut matcher = CepMatcher::new(10_000);
                    let pattern = CepPattern::new(
                        "test_pattern",
                        vec![
                            "event1".to_string(),
                            "event2".to_string(),
                            "event3".to_string(),
                        ],
                        60,
                    );
                    matcher.add_pattern(pattern);

                    let mut total_matches = 0;
                    for triple in &triples {
                        let matches = matcher.process(triple.clone());
                        total_matches += matches.len();
                    }
                    black_box(total_matches)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark streaming query window types
fn bench_window_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming/window_types");

    let window_configs = vec![
        ("tumbling", WindowConfig::tumbling(60)),
        ("sliding", WindowConfig::sliding(60, 10)),
        ("count", WindowConfig::count(100)),
        ("session", WindowConfig::session(30)),
        ("landmark", WindowConfig::landmark()),
    ];

    for (name, config) in window_configs {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut agg = WindowedAggregator::new(config.clone());
                for i in 0..1_000 {
                    agg.add(i as f64);
                }
                black_box(agg.count())
            });
        });
    }

    group.finish();
}

// ============================================================================
// PROPERTY GRAPH BRIDGE BENCHMARKS
// ============================================================================

/// Benchmark RDF-star to LPG conversion
fn bench_rdf_to_lpg(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_graph/rdf_to_lpg");

    for size in [10, 100, 1_000].iter() {
        let store = create_test_store(*size);
        let config = ConversionConfig::default();
        let bridge = PropertyGraphBridge::new(config);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let lpg = bridge.rdf_to_lpg(&store).unwrap();
                black_box(lpg.nodes.len())
            });
        });
    }

    group.finish();
}

/// Benchmark LPG to RDF-star conversion
fn bench_lpg_to_rdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_graph/lpg_to_rdf");

    for size in [10, 100, 1_000].iter() {
        let lpg = create_test_lpg(*size);
        let config = ConversionConfig::default();
        let bridge = PropertyGraphBridge::new(config);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let store = bridge.lpg_to_rdf(&lpg).unwrap();
                black_box(store.len())
            });
        });
    }

    group.finish();
}

/// Benchmark roundtrip conversion (RDF ↔ LPG)
fn bench_property_graph_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_graph/roundtrip");

    for size in [100, 1_000].iter() {
        let store = create_test_store(*size);
        let config = ConversionConfig::default();
        let bridge = PropertyGraphBridge::new(config);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                // RDF -> LPG
                let lpg = bridge.rdf_to_lpg(&store).unwrap();
                // LPG -> RDF
                let restored = bridge.lpg_to_rdf(&lpg).unwrap();
                black_box(restored.len())
            });
        });
    }

    group.finish();
}

/// Benchmark Cypher script generation
fn bench_cypher_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_graph/cypher_generation");

    for size in [10, 100, 1_000].iter() {
        let lpg = create_test_lpg(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let script = lpg.to_cypher_script().unwrap();
                black_box(script.len())
            });
        });
    }

    group.finish();
}

/// Benchmark property graph with quoted triples
fn bench_lpg_quoted_triples(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_graph/quoted_triples");

    for quoted_ratio in [0, 25, 50, 100].iter() {
        let store = create_quoted_triple_store(1_000, *quoted_ratio);
        let config = ConversionConfig {
            quoted_as_edge_properties: true,
            ..ConversionConfig::default()
        };
        let bridge = PropertyGraphBridge::new(config);

        group.bench_with_input(
            BenchmarkId::new("conversion", format!("{}%_quoted", quoted_ratio)),
            quoted_ratio,
            |b, &_ratio| {
                b.iter(|| {
                    let lpg = bridge.rdf_to_lpg(&store).unwrap();
                    black_box(lpg.edges.len())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// INTEGRATED BENCHMARKS (combining v0.3.0 features)
// ============================================================================

/// Benchmark HDT-star with property graph conversion
fn bench_hdt_lpg_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("integration/hdt_lpg");

    let store = create_test_store(1_000);

    group.bench_function("hdt_encode_lpg_convert", |b| {
        b.iter(|| {
            // Convert to HDT-star
            let config = HdtStarConfig::default();
            let mut builder = HdtStarBuilder::new(config);
            builder.add_store(&store).unwrap();

            let mut buffer = Vec::new();
            builder.write(&mut buffer).unwrap();

            // Decode from HDT-star
            let mut cursor = Cursor::new(&buffer);
            let reader = HdtStarReader::read(&mut cursor).unwrap();
            let restored = reader.to_store().unwrap();

            // Convert to property graph
            let pg_config = ConversionConfig::default();
            let bridge = PropertyGraphBridge::new(pg_config);
            let lpg = bridge.rdf_to_lpg(&restored).unwrap();

            black_box(lpg.nodes.len())
        });
    });

    group.finish();
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a test store with specified number of triples
fn create_test_store(size: usize) -> StarStore {
    let store = StarStore::new();

    for i in 0..size {
        let s = format!("http://example.org/subject{}", i);
        let p = format!("http://example.org/predicate{}", i % 10);
        let o = format!("Value {}", i);

        let triple = StarTriple::new(
            StarTerm::iri(&s).unwrap(),
            StarTerm::iri(&p).unwrap(),
            StarTerm::literal(&o).unwrap(),
        );
        store.insert(&triple).unwrap();
    }

    store
}

/// Create a test store with quoted triples
fn create_quoted_triple_store(size: usize, quoted_percentage: usize) -> StarStore {
    let store = StarStore::new();
    let quoted_count = (size * quoted_percentage) / 100;

    // Regular triples
    for i in 0..(size - quoted_count) {
        let s = format!("http://example.org/subject{}", i);
        let triple = StarTriple::new(
            StarTerm::iri(&s).unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::literal(&format!("Value {}", i)).unwrap(),
        );
        store.insert(&triple).unwrap();
    }

    // Quoted triples
    for i in 0..quoted_count {
        let base = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/s{}", i)).unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri(&format!("http://example.org/o{}", i)).unwrap(),
        );

        let quoted = StarTriple::new(
            StarTerm::quoted_triple(base),
            StarTerm::iri("http://example.org/meta").unwrap(),
            StarTerm::literal("metadata").unwrap(),
        );
        store.insert(&quoted).unwrap();
    }

    store
}

/// Create a test event stream for CEP benchmarks
fn create_event_stream(count: usize) -> Vec<StarTriple> {
    let mut triples = Vec::new();

    let events = ["event1", "event2", "event3", "other"];

    for i in 0..count {
        let event = events[i % events.len()];
        let triple = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/event{}", i)).unwrap(),
            StarTerm::iri(&format!("http://example.org/{}", event)).unwrap(),
            StarTerm::literal(&format!("data{}", i)).unwrap(),
        );
        triples.push(triple);
    }

    triples
}

/// Create a test labeled property graph
fn create_test_lpg(size: usize) -> LabeledPropertyGraph {
    let mut lpg = LabeledPropertyGraph::new();

    // Create nodes
    for i in 0..size {
        let mut node = LpgNode::new(format!("n{}", i));
        node.add_label("Person");
        node.set_property("name", PropertyValue::String(format!("Person{}", i)));
        node.set_property("age", PropertyValue::Integer(20 + (i % 50) as i64));
        lpg.add_node(node);
    }

    // Create edges
    for i in 0..(size / 2) {
        let mut edge = LpgEdge::new(format!("n{}", i), "knows", format!("n{}", (i + 1) % size));
        edge.set_property("since", PropertyValue::Integer(2020 + (i % 5) as i64));
        lpg.add_edge(edge);
    }

    lpg
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    name = hdt_star_benches;
    config = Criterion::default().sample_size(50);
    targets =
        bench_hdt_encoding,
        bench_hdt_decoding,
        bench_hdt_compression,
        bench_hdt_roundtrip,
        bench_hdt_quoted_triples
);

criterion_group!(
    name = streaming_benches;
    config = Criterion::default().sample_size(50);
    targets =
        bench_windowed_aggregation,
        bench_cep_pattern_matching,
        bench_window_operations
);

criterion_group!(
    name = property_graph_benches;
    config = Criterion::default().sample_size(50);
    targets =
        bench_rdf_to_lpg,
        bench_lpg_to_rdf,
        bench_property_graph_roundtrip,
        bench_cypher_generation,
        bench_lpg_quoted_triples
);

criterion_group!(
    name = integration_benches;
    config = Criterion::default().sample_size(30);
    targets =
        bench_hdt_lpg_integration
);

criterion_main!(
    hdt_star_benches,
    streaming_benches,
    property_graph_benches,
    integration_benches
);
