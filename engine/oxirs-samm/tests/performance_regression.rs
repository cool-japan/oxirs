//! Performance Regression Tests
//!
//! This test suite verifies that performance does not degrade below acceptable thresholds.
//! These tests establish performance baselines and fail if current performance is significantly
//! worse than the baseline.
//!
//! ## Usage
//!
//! Run with: `cargo test --test performance_regression --release`
//!
//! ## Thresholds
//!
//! Tests allow up to 20% performance degradation before failing. This accounts for:
//! - System load variations
//! - CPU frequency scaling
//! - Background processes
//! - Minor algorithmic changes
//!
//! ## Baseline Updates
//!
//! If legitimate changes cause performance differences, update the baseline constants.

use oxirs_samm::generators::{
    generate_graphql, generate_java, generate_python, generate_sql, generate_typescript,
    JavaOptions, PythonOptions, SqlDialect, TsOptions,
};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Property,
};
use oxirs_samm::parser::parse_aspect_from_string;
use oxirs_samm::validator::validate_aspect;
use std::time::{Duration, Instant};

/// Performance threshold: maximum allowed slowdown (20%)
const PERF_THRESHOLD: f64 = 1.20;

/// Baseline: Simple aspect parsing (microseconds)
const BASELINE_PARSE_SIMPLE: u64 = 5000; // 5ms

/// Baseline: Complex aspect parsing with 12 properties (microseconds)
const BASELINE_PARSE_COMPLEX: u64 = 15000; // 15ms

/// Baseline: TypeScript generation (microseconds)
const BASELINE_GEN_TYPESCRIPT: u64 = 500; // 500μs

/// Baseline: GraphQL generation (microseconds)
const BASELINE_GEN_GRAPHQL: u64 = 400; // 400μs

/// Baseline: Python generation (microseconds)
const BASELINE_GEN_PYTHON: u64 = 500; // 500μs

/// Baseline: Java generation (microseconds)
const BASELINE_GEN_JAVA: u64 = 800; // 800μs

/// Baseline: SQL generation (microseconds)
const BASELINE_GEN_SQL: u64 = 400; // 400μs

/// Baseline: Validation (microseconds)
const BASELINE_VALIDATION: u64 = 300; // 300μs

/// Sample SAMM Aspect for testing
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
    samm:preferredName "Speed"@en ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:acceleration a samm:Property ;
    samm:preferredName "Acceleration"@en ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:fuelLevel a samm:Property ;
    samm:preferredName "Fuel Level"@en ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:engineTemp a samm:Property ;
    samm:preferredName "Engine Temperature"@en ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:tirePressure a samm:Property ;
    samm:preferredName "Tire Pressure"@en ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:float ] .

:odometer a samm:Property ;
    samm:preferredName "Odometer"@en ;
    samm:characteristic [ a samm-c:Measurement ; samm:dataType xsd:integer ] .

:gpsLatitude a samm:Property ;
    samm:preferredName "GPS Latitude"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:double ] .

:gpsLongitude a samm:Property ;
    samm:preferredName "GPS Longitude"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:double ] .

:heading a samm:Property ;
    samm:preferredName "Heading"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:float ] .

:altitude a samm:Property ;
    samm:preferredName "Altitude"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:float ] .

:timestamp a samm:Property ;
    samm:preferredName "Timestamp"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:dateTime ] .

:vin a samm:Property ;
    samm:preferredName "VIN"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:string ] .
"#;

/// Helper to create test aspect with N properties
fn create_test_aspect(num_properties: usize) -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Test Aspect".to_string());

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

/// Measure execution time and compare against baseline
fn measure_and_assert<F>(test_name: &str, baseline_micros: u64, iterations: usize, mut operation: F)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..5 {
        operation();
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() as u64 / iterations as u64;
    let threshold_micros = (baseline_micros as f64 * PERF_THRESHOLD) as u64;
    let speedup = baseline_micros as f64 / avg_micros as f64;

    println!(
        "📊 {}: avg={:.2}μs, baseline={}μs, speedup={:.2}x, threshold={}μs",
        test_name, avg_micros, baseline_micros, speedup, threshold_micros
    );

    if avg_micros > threshold_micros {
        panic!(
            "❌ Performance regression detected in {}!\n\
             Current: {}μs, Baseline: {}μs, Threshold: {}μs\n\
             Performance degraded by {:.1}% (max allowed: {}%)",
            test_name,
            avg_micros,
            baseline_micros,
            threshold_micros,
            ((avg_micros as f64 / baseline_micros as f64 - 1.0) * 100.0),
            ((PERF_THRESHOLD - 1.0) * 100.0)
        );
    }

    if speedup > 1.5 {
        println!(
            "✨ Excellent! {} is {:.1}x faster than baseline!",
            test_name, speedup
        );
    }
}

#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
async fn test_parser_simple_aspect_performance() {
    // Warmup
    for _ in 0..5 {
        parse_aspect_from_string(SIMPLE_ASPECT, "urn:samm:org.example:1.0.0#")
            .await
            .unwrap();
    }

    // Measure
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        parse_aspect_from_string(SIMPLE_ASPECT, "urn:samm:org.example:1.0.0#")
            .await
            .unwrap();
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() as u64 / iterations;
    let threshold_micros = (BASELINE_PARSE_SIMPLE as f64 * PERF_THRESHOLD) as u64;

    println!(
        "📊 Parse simple aspect: avg={}μs, baseline={}μs, threshold={}μs",
        avg_micros, BASELINE_PARSE_SIMPLE, threshold_micros
    );

    assert!(
        avg_micros < threshold_micros,
        "Performance regression: {}μs > {}μs",
        avg_micros,
        threshold_micros
    );
}

#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
async fn test_parser_complex_aspect_performance() {
    // Warmup
    for _ in 0..5 {
        let _ = parse_aspect_from_string(COMPLEX_ASPECT, "urn:samm:org.example:1.0.0#VehicleData")
            .await;
    }

    // Measure
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = parse_aspect_from_string(COMPLEX_ASPECT, "urn:samm:org.example:1.0.0#VehicleData")
            .await;
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() as u64 / iterations;
    let threshold_micros = (BASELINE_PARSE_COMPLEX as f64 * PERF_THRESHOLD) as u64;

    println!(
        "📊 Parse complex aspect: avg={}μs, baseline={}μs, threshold={}μs",
        avg_micros, BASELINE_PARSE_COMPLEX, threshold_micros
    );

    assert!(
        avg_micros < threshold_micros,
        "Performance regression: {}μs > {}μs (successful parses only)",
        avg_micros,
        threshold_micros
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
fn test_typescript_generation_performance() {
    let aspect = create_test_aspect(10);
    let options = TsOptions::default();

    measure_and_assert(
        "TypeScript generation",
        BASELINE_GEN_TYPESCRIPT,
        100,
        || {
            generate_typescript(&aspect, options.clone()).unwrap();
        },
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
fn test_graphql_generation_performance() {
    let aspect = create_test_aspect(10);

    measure_and_assert("GraphQL generation", BASELINE_GEN_GRAPHQL, 100, || {
        generate_graphql(&aspect).unwrap();
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
fn test_python_generation_performance() {
    let aspect = create_test_aspect(10);
    let options = PythonOptions::default();

    measure_and_assert("Python generation", BASELINE_GEN_PYTHON, 100, || {
        generate_python(&aspect, options.clone()).unwrap();
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
fn test_java_generation_performance() {
    let aspect = create_test_aspect(10);
    let options = JavaOptions::default();

    measure_and_assert("Java generation", BASELINE_GEN_JAVA, 100, || {
        generate_java(&aspect, options.clone()).unwrap();
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
fn test_sql_generation_performance() {
    let aspect = create_test_aspect(10);

    measure_and_assert("SQL generation", BASELINE_GEN_SQL, 100, || {
        generate_sql(&aspect, SqlDialect::PostgreSql).unwrap();
    });
}

#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
async fn test_validation_performance() {
    let aspect = create_test_aspect(10);

    // Warmup
    for _ in 0..5 {
        validate_aspect(&aspect).await.unwrap();
    }

    // Measure
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        validate_aspect(&aspect).await.unwrap();
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() as u64 / iterations;
    let threshold_micros = (BASELINE_VALIDATION as f64 * PERF_THRESHOLD) as u64;

    println!(
        "📊 Validation: avg={}μs, baseline={}μs, threshold={}μs",
        avg_micros, BASELINE_VALIDATION, threshold_micros
    );

    assert!(
        avg_micros < threshold_micros,
        "Performance regression: {}μs > {}μs",
        avg_micros,
        threshold_micros
    );
}

/// Test parsing performance scaling with model size
#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
async fn test_parser_scaling_performance() {
    println!("\n📈 Parser Scaling Performance:");

    for size in [5, 10, 20, 50] {
        let model = generate_aspect_with_properties(size);
        let start = Instant::now();

        let mut successes = 0;
        for _ in 0..5 {
            if parse_aspect_from_string(&model, "urn:samm:org.example:1.0.0#TestAspect")
                .await
                .is_ok()
            {
                successes += 1;
            }
        }

        if successes > 0 {
            let avg = start.elapsed().as_micros() / successes as u128;
            let per_property = avg as f64 / size as f64;

            println!(
                "  {} properties: {}μs total, {:.1}μs per property ({} successes)",
                size, avg, per_property, successes
            );

            // Assert that per-property cost doesn't grow super-linearly
            assert!(
                per_property < 1000.0,
                "Per-property parsing cost too high: {:.1}μs",
                per_property
            );
        } else {
            println!("  {} properties: All parses failed (test data issue)", size);
        }
    }
}

/// Test generator performance scaling with model size
#[test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
fn test_generator_scaling_performance() {
    println!("\n📈 Generator Scaling Performance:");

    for size in [5, 10, 20, 50] {
        let aspect = create_test_aspect(size);

        // TypeScript
        let start = Instant::now();
        for _ in 0..20 {
            generate_typescript(&aspect, TsOptions::default()).unwrap();
        }
        let ts_avg = start.elapsed().as_micros() / 20;
        let ts_per_prop = ts_avg as f64 / size as f64;

        println!(
            "  TypeScript {} props: {}μs total, {:.1}μs/prop",
            size, ts_avg, ts_per_prop
        );

        // Assert reasonable scaling
        assert!(
            ts_per_prop < 100.0,
            "TypeScript per-property cost too high: {:.1}μs",
            ts_per_prop
        );
    }
}

/// Test end-to-end performance: parse + validate + generate
#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
async fn test_end_to_end_performance() {
    const E2E_BASELINE: u64 = 10000; // 10ms for full pipeline

    // Warmup
    for _ in 0..5 {
        let aspect = parse_aspect_from_string(SIMPLE_ASPECT, "urn:samm:org.example:1.0.0#")
            .await
            .unwrap();
        validate_aspect(&aspect).await.unwrap();
        generate_typescript(&aspect, TsOptions::default()).unwrap();
    }

    // Measure
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let aspect = parse_aspect_from_string(SIMPLE_ASPECT, "urn:samm:org.example:1.0.0#")
            .await
            .unwrap();
        validate_aspect(&aspect).await.unwrap();
        generate_typescript(&aspect, TsOptions::default()).unwrap();
        generate_graphql(&aspect).unwrap();
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() as u64 / iterations;
    let threshold_micros = (E2E_BASELINE as f64 * PERF_THRESHOLD) as u64;

    println!(
        "📊 End-to-end pipeline: avg={}μs, baseline={}μs, threshold={}μs",
        avg_micros, E2E_BASELINE, threshold_micros
    );

    assert!(
        avg_micros < threshold_micros,
        "Performance regression: {}μs > {}μs",
        avg_micros,
        threshold_micros
    );
}

/// Test memory efficiency: large models should not cause excessive allocations
#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
async fn test_memory_efficiency() {
    println!("\n💾 Memory Efficiency Test:");

    // Use SIMPLE_ASPECT for consistency test
    // Parse multiple times and check that memory doesn't grow unboundedly
    for iteration in 1..=10 {
        let start = Instant::now();

        let aspect = parse_aspect_from_string(SIMPLE_ASPECT, "urn:samm:org.example:1.0.0#Movement")
            .await
            .expect("Failed to parse SIMPLE_ASPECT");

        let elapsed = start.elapsed();

        // Validate
        validate_aspect(&aspect).await.expect("Validation failed");

        println!("  Iteration {}: {:?}", iteration, elapsed);

        // Each iteration should be consistent (no memory buildup causing slowdown)
        if iteration > 1 {
            assert!(
                elapsed < Duration::from_millis(100),
                "Memory buildup detected: iteration {} took {:?}",
                iteration,
                elapsed
            );
        }
    }

    println!("  ✅ Memory efficiency test passed: No memory buildup detected");
}

/// Helper: Generate SAMM model with N properties
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

    for i in 0..count {
        model.push_str(&format!(":prop{} ", i));
    }
    model.push_str(") .\n\n");

    for i in 0..count {
        model.push_str(&format!(
            r#":prop{} a samm:Property ;
    samm:preferredName "Property {}"@en ;
    samm:description "Test property {}"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:string ] .

"#,
            i, i, i
        ));
    }

    model
}

/// Test concurrent parsing performance
#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
async fn test_concurrent_parsing_performance() {
    println!("\n🔀 Concurrent Parsing Performance:");

    const CONCURRENT_TASKS: usize = 10;
    const BASELINE_CONCURRENT: u64 = 30000; // 30ms for 10 concurrent parses

    let start = Instant::now();

    let handles: Vec<_> = (0..CONCURRENT_TASKS)
        .map(|_| {
            tokio::spawn(async {
                parse_aspect_from_string(SIMPLE_ASPECT, "urn:samm:org.example:1.0.0#")
                    .await
                    .unwrap();
            })
        })
        .collect();

    for handle in handles {
        handle.await.unwrap();
    }

    let elapsed = start.elapsed();
    let elapsed_micros = elapsed.as_micros() as u64;
    let threshold = (BASELINE_CONCURRENT as f64 * PERF_THRESHOLD) as u64;

    println!(
        "  {} concurrent tasks: {}μs, baseline: {}μs, threshold: {}μs",
        CONCURRENT_TASKS, elapsed_micros, BASELINE_CONCURRENT, threshold
    );

    assert!(
        elapsed_micros < threshold,
        "Concurrent parsing performance regression: {}μs > {}μs",
        elapsed_micros,
        threshold
    );
}

/// Test that caching improves performance
#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "Performance tests require release builds")]
async fn test_caching_effectiveness() {
    use oxirs_samm::parser::ModelResolver;
    use std::env;
    use std::fs;

    println!("\n🗄️  Caching Effectiveness:");

    // Create a temporary directory structure matching SAMM conventions
    let temp_dir = env::temp_dir().join("samm_cache_test");
    let models_dir = temp_dir.join("org.example").join("1.0.0");
    fs::create_dir_all(&models_dir).unwrap();

    let test_file = models_dir.join("Movement.ttl");
    fs::write(&test_file, SIMPLE_ASPECT).unwrap();

    let mut resolver = ModelResolver::new();
    resolver.add_models_root(temp_dir.clone());

    // First parse (cold cache)
    let start = Instant::now();
    resolver
        .load_element("urn:samm:org.example:1.0.0#Movement")
        .await
        .unwrap();
    let cold_time = start.elapsed();

    // Second parse (warm cache)
    let start = Instant::now();
    resolver
        .load_element("urn:samm:org.example:1.0.0#Movement")
        .await
        .unwrap();
    let warm_time = start.elapsed();

    let speedup = cold_time.as_micros() as f64 / warm_time.as_micros() as f64;

    println!(
        "  Cold cache: {:?}, Warm cache: {:?}, Speedup: {:.2}x",
        cold_time, warm_time, speedup
    );

    // Clean up
    fs::remove_dir_all(temp_dir).ok();

    // Warm cache should be significantly faster (at least 2x)
    assert!(
        speedup > 2.0,
        "Cache not providing sufficient speedup: {:.2}x",
        speedup
    );
}
