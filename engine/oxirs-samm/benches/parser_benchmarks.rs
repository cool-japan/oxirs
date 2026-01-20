//! Benchmarks for SAMM parser performance
//!
//! These benchmarks measure the performance of parsing SAMM models from Turtle format.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_samm::parser::parse_aspect_from_string;
use std::time::Duration;

/// Sample SAMM Aspect model for benchmarking
const SIMPLE_ASPECT: &str = r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:samm:org.example:1.0.0#> .

:Movement a samm:Aspect ;
    samm:preferredName "Movement"@en ;
    samm:description "Movement of a vehicle"@en ;
    samm:properties ( :speed :position ) .

:speed a samm:Property ;
    samm:preferredName "Speed"@en ;
    samm:description "Speed of the vehicle"@en ;
    samm:characteristic :SpeedCharacteristic .

:SpeedCharacteristic a samm-c:Measurement ;
    samm:dataType xsd:float ;
    samm-c:unit <urn:samm:org.eclipse.esmf.samm:unit:2.3.0#kilometrePerHour> .

:position a samm:Property ;
    samm:preferredName "Position"@en ;
    samm:description "GPS position"@en ;
    samm:characteristic :PositionCharacteristic .

:PositionCharacteristic a samm:Characteristic ;
    samm:dataType xsd:string .
"#;

/// Complex SAMM Aspect with many properties
const COMPLEX_ASPECT: &str = r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:samm:org.example:1.0.0#> .

:VehicleData a samm:Aspect ;
    samm:preferredName "Vehicle Data"@en ;
    samm:description "Comprehensive vehicle telemetry data"@en ;
    samm:properties (
        :speed :acceleration :fuelLevel :engineTemp
        :tirePressure :odometer :gpsLatitude :gpsLongitude
        :heading :altitude :timestamp :vin
    ) .

:speed a samm:Property ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:acceleration a samm:Property ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:fuelLevel a samm:Property ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:engineTemp a samm:Property ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:tirePressure a samm:Property ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:odometer a samm:Property ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:integer ] .

:gpsLatitude a samm:Property ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:double ] .

:gpsLongitude a samm:Property ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:double ] .

:heading a samm:Property ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:float ] .

:altitude a samm:Property ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:float ] .

:timestamp a samm:Property ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:dateTime ] .

:vin a samm:Property ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:string ] .
"#;

fn bench_parse_simple_aspect(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("parse_simple_aspect", |b| {
        b.to_async(&runtime).iter(|| async {
            let result =
                parse_aspect_from_string(black_box(SIMPLE_ASPECT), "urn:samm:org.example:1.0.0#")
                    .await;
            result.unwrap()
        });
    });
}

fn bench_parse_complex_aspect(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("parse_complex_aspect", |b| {
        b.to_async(&runtime).iter(|| async {
            let result =
                parse_aspect_from_string(black_box(COMPLEX_ASPECT), "urn:samm:org.example:1.0.0#")
                    .await;
            result.unwrap()
        });
    });
}

fn bench_parse_scaling(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("parse_scaling");

    for size in [1, 5, 10, 20].iter() {
        // Generate model with N properties
        let model = generate_aspect_with_properties(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_properties", size)),
            &model,
            |b, model| {
                b.to_async(&runtime).iter(|| async {
                    let result =
                        parse_aspect_from_string(black_box(model), "urn:samm:org.example:1.0.0#")
                            .await;
                    result.unwrap()
                });
            },
        );
    }

    group.finish();
}

fn generate_aspect_with_properties(count: usize) -> String {
    let mut model = String::from(
        r#"@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:samm:org.example:1.0.0#> .

:TestAspect a samm:Aspect ;
    samm:preferredName "Test Aspect"@en ;
    samm:properties ( "#,
    );

    // Add property references
    for i in 0..count {
        model.push_str(&format!(":prop{} ", i));
    }
    model.push_str(") .\n\n");

    // Add property definitions
    for i in 0..count {
        model.push_str(&format!(
            r#":prop{} a samm:Property ;
    samm:preferredName "Property {}"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:string ] .

"#,
            i, i
        ));
    }

    model
}

criterion_group! {
    name = parser_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets = bench_parse_simple_aspect, bench_parse_complex_aspect, bench_parse_scaling
}

criterion_main!(parser_benches);
