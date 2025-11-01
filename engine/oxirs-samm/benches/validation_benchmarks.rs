//! Benchmarks for SAMM validation performance
//!
//! These benchmarks measure the performance of validating SAMM models.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Property,
};
use oxirs_samm::validator::ShaclValidator;
use std::time::Duration;

/// Create a valid test aspect
fn create_valid_aspect(num_properties: usize) -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Test Aspect".to_string());
    aspect
        .metadata
        .add_description("en".to_string(), "A test aspect".to_string());

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

/// Create an aspect with validation issues
fn create_invalid_aspect() -> Aspect {
    let mut aspect = Aspect::new("invalid-urn".to_string()); // Invalid URN format
                                                             // Missing preferred name
    aspect
        .metadata
        .add_description("en".to_string(), "Invalid aspect".to_string());

    // Property without characteristic
    let prop = Property::new("urn:samm:org.example:1.0.0#prop1".to_string());
    // Note: prop.characteristic is None by default, which violates validation rules
    aspect.add_property(prop);

    aspect
}

fn bench_validate_simple_aspect(c: &mut Criterion) {
    let aspect = create_valid_aspect(5);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let validator = ShaclValidator::new();

    c.bench_function("validate_simple_aspect", |b| {
        b.to_async(&runtime)
            .iter(|| async { validator.validate(black_box(&aspect)).await.unwrap() });
    });
}

fn bench_validate_complex_aspect(c: &mut Criterion) {
    let aspect = create_valid_aspect(20);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let validator = ShaclValidator::new();

    c.bench_function("validate_complex_aspect", |b| {
        b.to_async(&runtime)
            .iter(|| async { validator.validate(black_box(&aspect)).await.unwrap() });
    });
}

fn bench_validate_invalid_aspect(c: &mut Criterion) {
    let aspect = create_invalid_aspect();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let validator = ShaclValidator::new();

    c.bench_function("validate_invalid_aspect", |b| {
        b.to_async(&runtime).iter(|| async {
            // This will return errors, but that's expected
            validator.validate(black_box(&aspect)).await.unwrap()
        });
    });
}

fn bench_validation_scaling(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("validation_scaling");
    let validator = ShaclValidator::new();

    for size in [5, 10, 20, 50].iter() {
        let aspect = create_valid_aspect(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_properties", size)),
            &aspect,
            |b, aspect| {
                b.to_async(&runtime)
                    .iter(|| async { validator.validate(black_box(aspect)).await.unwrap() });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = validation_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets =
        bench_validate_simple_aspect,
        bench_validate_complex_aspect,
        bench_validate_invalid_aspect,
        bench_validation_scaling
}

criterion_main!(validation_benches);
