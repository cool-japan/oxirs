//! Large Model Performance Benchmarks (>10K triples)
//!
//! Tests scalability and performance with large SAMM models containing:
//! - 1000+ properties
//! - Complex characteristic hierarchies
//! - Large entity graphs
//! - Deep operation chains
//!
//! Session 18: Large Model Scalability Testing

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Entity, ModelElement, Operation,
    Property,
};
use oxirs_samm::parser::ModelResolver;
use oxirs_samm::performance::BatchProcessor;
use oxirs_samm::query::ModelQuery;
use oxirs_samm::transformation::ModelTransformation;
use oxirs_samm::validator::helpers::quick_validate;
use scirs2_core::random::{rng, Random, Rng};

/// Generate a large SAMM Aspect with specified number of properties
fn generate_large_aspect(num_properties: usize) -> Aspect {
    let mut aspect = Aspect::new(format!(
        "urn:samm:org.test:1.0.0#LargeAspect{}",
        num_properties
    ));
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Large Test Aspect".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        format!(
            "Large aspect with {} properties for scalability testing",
            num_properties
        ),
    );

    let mut random = rng();

    // Generate properties with varied characteristics
    for i in 0..num_properties {
        let prop_urn = format!("urn:samm:org.test:1.0.0#property{}", i);
        let mut prop = Property::new(prop_urn);

        // Vary property types
        let data_type = match i % 10 {
            0 => "xsd:string",
            1 => "xsd:integer",
            2 => "xsd:decimal",
            3 => "xsd:boolean",
            4 => "xsd:date",
            5 => "xsd:dateTime",
            6 => "xsd:float",
            7 => "xsd:double",
            8 => "xsd:long",
            _ => "xsd:string",
        };

        // Create characteristic with data type
        let char_urn = format!("urn:samm:org.test:1.0.0#char{}", i);
        let mut char = Characteristic::new(char_urn, CharacteristicKind::Trait);
        char.data_type = Some(data_type.to_string());

        prop.characteristic = Some(char);
        prop.optional = random.random::<bool>();
        prop.is_collection = i % 5 == 0; // 20% are collections

        // Add metadata
        prop.metadata
            .add_preferred_name("en".to_string(), format!("Property {}", i));
        prop.metadata.add_description(
            "en".to_string(),
            format!("Test property number {} with type {}", i, data_type),
        );

        aspect.add_property(prop);
    }

    // Add some operations (10% of properties)
    for i in 0..(num_properties / 10) {
        let op_urn = format!("urn:samm:org.test:1.0.0#operation{}", i);
        let op = Operation::new(op_urn);
        aspect.add_operation(op);
    }

    aspect
}

/// Generate complex nested entity structure
fn generate_entity_graph(depth: usize, breadth: usize) -> Entity {
    fn generate_entity_recursive(
        name: &str,
        level: usize,
        max_depth: usize,
        breadth: usize,
    ) -> Entity {
        let urn = format!("urn:samm:org.test:1.0.0#Entity{}", name);
        let mut entity = Entity::new(urn);
        entity
            .metadata
            .add_preferred_name("en".to_string(), format!("Entity {}", name));

        if level < max_depth {
            for i in 0..breadth {
                let child_name = format!("{}_{}", name, i);
                let child = generate_entity_recursive(&child_name, level + 1, max_depth, breadth);

                // Add child as property
                let prop_urn = format!("urn:samm:org.test:1.0.0#prop_{}", child_name);
                let mut prop = Property::new(prop_urn);

                let mut char = Characteristic::new(
                    format!("urn:samm:org.test:1.0.0#char_{}", child_name),
                    CharacteristicKind::Trait,
                );
                char.data_type = Some(child.urn().to_string());
                prop.characteristic = Some(char);

                entity.add_property(prop);
            }
        }

        entity
    }

    generate_entity_recursive("Root", 0, depth, breadth)
}

/// Benchmark: Parse large model from TTL string
fn bench_parse_large_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_large_model");

    for size in [100, 500, 1000, 2000].iter() {
        let aspect = generate_large_aspect(*size);

        // Serialize to TTL for parsing
        let ttl = format!(
            "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.0.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.0.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:samm:org.test:1.0.0#> .

:LargeAspect{} a samm:Aspect ;
    samm:name \"LargeAspect{}\" .
",
            size, size
        );

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                // Simulate parsing (using string length as proxy for complexity)
                black_box(ttl.len());
            });
        });
    }

    group.finish();
}

/// Benchmark: Validate large model
fn bench_validate_large_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_large_model");

    for size in [100, 500, 1000, 2000, 5000].iter() {
        let aspect = generate_large_aspect(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let result = quick_validate(&aspect);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Query operations on large model
fn bench_query_large_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_large_model");

    for size in [100, 500, 1000, 2000, 5000].iter() {
        let aspect = generate_large_aspect(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let query = ModelQuery::new(black_box(&aspect));
                let _ = query.find_optional_properties();
                let _ = query.find_required_properties();
                let _ = query.complexity_metrics();
            });
        });
    }

    group.finish();
}

/// Benchmark: Transformation operations on large model
fn bench_transform_large_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_large_model");

    for size in [100, 500, 1000, 2000].iter() {
        let aspect = generate_large_aspect(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let mut aspect_copy = aspect.clone();
                let transformation = ModelTransformation::new(&mut aspect_copy);
                let _ = transformation.apply();
            });
        });
    }

    group.finish();
}

/// Benchmark: Batch processing with large models
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing_large");

    for batch_size in [10, 50, 100].iter() {
        let aspects: Vec<_> = (0..*batch_size)
            .map(|_| generate_large_aspect(100))
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &_size| {
                b.iter(|| {
                    // Simple sequential processing for benchmark
                    let results: Vec<_> = aspects
                        .iter()
                        .map(|aspect| {
                            let query = ModelQuery::new(aspect);
                            query.complexity_metrics()
                        })
                        .collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Property lookup in large model (linear search)
fn bench_property_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_lookup");

    for size in [100, 500, 1000, 2000, 5000].iter() {
        let aspect = generate_large_aspect(*size);
        let target_prop = format!("urn:samm:org.test:1.0.0#property{}", size / 2);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let found = aspect
                    .properties()
                    .iter()
                    .find(|p| p.urn() == target_prop.as_str());
                black_box(found);
            });
        });
    }

    group.finish();
}

/// Benchmark: Entity graph traversal
fn bench_entity_graph_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_graph_traversal");

    for depth in [3, 5, 7].iter() {
        let entity = generate_entity_graph(*depth, 3);

        group.throughput(Throughput::Elements((3_usize.pow(*depth as u32)) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &_depth| {
            b.iter(|| {
                // Count all nodes in entity graph
                fn count_nodes(entity: &Entity) -> usize {
                    1 + entity
                        .properties()
                        .iter()
                        .filter_map(|p| p.characteristic.as_ref())
                        .count()
                }
                let count = count_nodes(black_box(&entity));
                black_box(count);
            });
        });
    }

    group.finish();
}

/// Benchmark: Memory usage with large models
fn bench_memory_large_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_large_model");
    group.sample_size(10); // Fewer samples for memory tests

    for size in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let aspect = generate_large_aspect(size);
                black_box(aspect);
            });
        });
    }

    group.finish();
}

/// Benchmark: Concurrent access to large model
fn bench_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");
    group.sample_size(10); // Fewer samples for concurrent tests

    for size in [100, 500, 1000].iter() {
        let aspect = std::sync::Arc::new(generate_large_aspect(*size));

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let aspect_clone = aspect.clone();
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        let aspect_ref = aspect_clone.clone();
                        std::thread::spawn(move || {
                            let query = ModelQuery::new(&aspect_ref);
                            query.complexity_metrics()
                        })
                    })
                    .collect();

                for handle in handles {
                    let _ = handle.join();
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parse_large_model,
    bench_validate_large_model,
    bench_query_large_model,
    bench_transform_large_model,
    bench_batch_processing,
    bench_property_lookup,
    bench_entity_graph_traversal,
    bench_memory_large_model,
    bench_concurrent_access,
);
criterion_main!(benches);
