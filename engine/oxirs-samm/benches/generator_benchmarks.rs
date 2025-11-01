//! Benchmarks for SAMM code generators
//!
//! These benchmarks measure the performance of generating code from SAMM models.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_samm::generators::{
    generate_graphql, generate_java, generate_python, generate_sql, generate_typescript,
    JavaOptions, PythonOptions, SqlDialect, TsOptions,
};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Property,
};
use std::time::Duration;

/// Create a test aspect with a given number of properties
fn create_test_aspect(num_properties: usize) -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Test Aspect".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "A test aspect for benchmarking".to_string(),
    );

    for i in 0..num_properties {
        let mut prop = Property::new(format!("urn:samm:org.example:1.0.0#prop{}", i))
            .with_characteristic(
                Characteristic::new(
                    format!("urn:samm:org.example:1.0.0#Char{}", i),
                    CharacteristicKind::Trait,
                )
                .with_data_type("xsd:string".to_string()),
            );

        prop.metadata
            .add_preferred_name("en".to_string(), format!("Property {}", i));
        aspect.add_property(prop);
    }

    aspect
}

fn bench_typescript_generation(c: &mut Criterion) {
    let aspect = create_test_aspect(10);
    let options = TsOptions::default();

    c.bench_function("generate_typescript", |b| {
        b.iter(|| generate_typescript(black_box(&aspect), black_box(options.clone())).unwrap());
    });
}

fn bench_graphql_generation(c: &mut Criterion) {
    let aspect = create_test_aspect(10);

    c.bench_function("generate_graphql", |b| {
        b.iter(|| generate_graphql(black_box(&aspect)).unwrap());
    });
}

fn bench_python_generation(c: &mut Criterion) {
    let aspect = create_test_aspect(10);
    let options = PythonOptions::default();

    c.bench_function("generate_python", |b| {
        b.iter(|| generate_python(black_box(&aspect), black_box(options.clone())).unwrap());
    });
}

fn bench_java_generation(c: &mut Criterion) {
    let aspect = create_test_aspect(10);
    let options = JavaOptions::default();

    c.bench_function("generate_java", |b| {
        b.iter(|| generate_java(black_box(&aspect), black_box(options.clone())).unwrap());
    });
}

fn bench_sql_generation(c: &mut Criterion) {
    let aspect = create_test_aspect(10);

    c.bench_function("generate_sql_postgresql", |b| {
        b.iter(|| generate_sql(black_box(&aspect), SqlDialect::PostgreSql).unwrap());
    });
}

fn bench_generator_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_scaling");

    for size in [5, 10, 20, 50].iter() {
        let aspect = create_test_aspect(*size);

        // TypeScript scaling
        group.bench_with_input(
            BenchmarkId::new("typescript", size),
            &aspect,
            |b, aspect| {
                b.iter(|| generate_typescript(black_box(aspect), TsOptions::default()).unwrap());
            },
        );

        // GraphQL scaling
        group.bench_with_input(BenchmarkId::new("graphql", size), &aspect, |b, aspect| {
            b.iter(|| generate_graphql(black_box(aspect)).unwrap());
        });

        // Python scaling
        group.bench_with_input(BenchmarkId::new("python", size), &aspect, |b, aspect| {
            b.iter(|| generate_python(black_box(aspect), PythonOptions::default()).unwrap());
        });
    }

    group.finish();
}

fn bench_all_generators(c: &mut Criterion) {
    let aspect = create_test_aspect(10);
    let mut group = c.benchmark_group("all_generators");

    group.bench_function("typescript", |b| {
        b.iter(|| generate_typescript(black_box(&aspect), TsOptions::default()).unwrap());
    });

    group.bench_function("graphql", |b| {
        b.iter(|| generate_graphql(black_box(&aspect)).unwrap());
    });

    group.bench_function("python", |b| {
        b.iter(|| generate_python(black_box(&aspect), PythonOptions::default()).unwrap());
    });

    group.bench_function("java", |b| {
        b.iter(|| generate_java(black_box(&aspect), JavaOptions::default()).unwrap());
    });

    group.bench_function("sql", |b| {
        b.iter(|| generate_sql(black_box(&aspect), SqlDialect::PostgreSql).unwrap());
    });

    group.finish();
}

criterion_group! {
    name = generator_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets =
        bench_typescript_generation,
        bench_graphql_generation,
        bench_python_generation,
        bench_java_generation,
        bench_sql_generation,
        bench_generator_scaling,
        bench_all_generators
}

criterion_main!(generator_benches);
