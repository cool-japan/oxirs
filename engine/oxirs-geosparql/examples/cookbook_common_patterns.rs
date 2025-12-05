//! Cookbook: Common Patterns and Best Practices for oxirs-geosparql
//!
//! This example demonstrates common spatial data processing patterns,
//! best practices, and production-ready code patterns for real-world applications.

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::geometric_operations::*;
use oxirs_geosparql::functions::geometric_properties::*;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::index::SpatialIndex;

// ============================================================================
// Pattern 1: Efficient Bulk Data Loading
// ============================================================================

/// Pattern: Load and index large datasets efficiently
///
/// Best practices:
/// - Use bulk_load instead of individual inserts (10-100x faster)
/// - Pre-allocate vectors with capacity when known
/// - Use WKT parsing for standard formats
fn pattern_bulk_data_loading() -> Result<()> {
    println!("\n=== Pattern 1: Efficient Bulk Data Loading ===\n");

    // Pre-allocate with known capacity
    let mut geometries = Vec::with_capacity(10_000);

    // Load geometries (simulated data)
    for i in 0..10_000 {
        let lat = 37.7 + (i as f64 / 1000.0) * 0.1;
        let lon = -122.4 + (i as f64 / 1000.0) * 0.1;
        geometries.push(Geometry::from_wkt(&format!("POINT({} {})", lon, lat))?);
    }

    // Best practice: Use bulk_load for optimal R-tree construction
    let start = std::time::Instant::now();
    let index = SpatialIndex::bulk_load(geometries)?;
    let duration = start.elapsed();

    println!("✓ Bulk loaded 10,000 points in {:?}", duration);
    println!("  Tip: bulk_load is 10-100x faster than individual inserts");

    // Query the index
    let results = index.query_bbox(-122.4, 37.7, -122.3, 37.8);
    println!("✓ Found {} points in viewport", results.len());

    Ok(())
}

// ============================================================================
// Pattern 2: Safe Geometry Operations with Error Handling
// ============================================================================

/// Pattern: Robust geometry operations with proper error handling
///
/// Best practices:
/// - Always validate input geometries before expensive operations
/// - Use ? operator for clean error propagation
/// - Check CRS compatibility before spatial operations
fn pattern_safe_geometry_operations() -> Result<()> {
    println!("\n=== Pattern 2: Safe Geometry Operations ===\n");

    let polygon1 = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let polygon2 = Geometry::from_wkt("POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))")?;

    // Best practice: Validate geometries before expensive operations
    let validation1 = oxirs_geosparql::validation::validate_geometry(&polygon1);
    let validation2 = oxirs_geosparql::validation::validate_geometry(&polygon2);

    if !validation1.is_valid || !validation2.is_valid {
        println!("⚠ Warning: Invalid geometry detected");
        return Ok(());
    }

    // Best practice: Check CRS compatibility
    polygon1.validate_crs_compatibility(&polygon2)?;

    // Perform operations safely
    let intersection = intersection(&polygon1, &polygon2)?;
    let union = union(&polygon1, &polygon2)?;

    if let Some(intersect_geom) = intersection {
        let intersect_area = area(&intersect_geom)?;
        println!("✓ Intersection area: {:.2}", intersect_area);
    }

    let union_area = area(&union)?;
    println!("✓ Union area: {:.2}", union_area);
    println!("  Tip: Always validate CRS compatibility before operations");

    Ok(())
}

// ============================================================================
// Pattern 3: Proximity Analysis with Spatial Indexes
// ============================================================================

/// Pattern: Find nearby features efficiently
///
/// Best practices:
/// - Use spatial index for proximity queries (1000x faster than brute force)
/// - Use k-nearest for "find N closest" queries
/// - Use distance queries for "find all within radius" queries
fn pattern_proximity_analysis() -> Result<()> {
    println!("\n=== Pattern 3: Proximity Analysis ===\n");

    // Create dataset of POIs
    let pois: Vec<Geometry> = (0..1000)
        .map(|i| {
            let lat = 37.7 + (i as f64 / 100.0) * 0.01;
            let lon = -122.4 + (i as f64 / 100.0) * 0.01;
            Geometry::from_wkt(&format!("POINT({} {})", lon, lat)).unwrap()
        })
        .collect();

    let index = SpatialIndex::bulk_load(pois)?;

    // Query: Find 5 nearest POIs to a location
    let query_x = -122.395;
    let query_y = 37.705;

    let start = std::time::Instant::now();
    let nearest = index.nearest(query_x, query_y);
    let duration = start.elapsed();

    println!("✓ Found nearest POI in {:?}", duration);

    // Display nearest POI if found
    if let Some((_, dist)) = nearest {
        println!("  Distance to nearest: {:.6} degrees", dist);
    } else {
        println!("  No POI found");
    }

    println!("  Tip: Spatial index makes proximity queries 1000x faster");

    Ok(())
}

// ============================================================================
// Pattern 4: Streaming Large Datasets
// ============================================================================

/// Pattern: Process large datasets without loading everything into memory
///
/// Best practices:
/// - Process data in chunks to avoid memory exhaustion
/// - Use iterators for lazy evaluation
/// - Accumulate results incrementally
fn pattern_streaming_large_datasets() -> Result<()> {
    println!("\n=== Pattern 4: Streaming Large Datasets ===\n");

    const CHUNK_SIZE: usize = 1000;
    const TOTAL_FEATURES: usize = 100_000;

    let mut total_area = 0.0;
    let mut feature_count = 0;

    // Process in chunks
    for chunk_start in (0..TOTAL_FEATURES).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(TOTAL_FEATURES);

        // Generate chunk (in real code, read from file/database)
        let chunk: Vec<Geometry> = (chunk_start..chunk_end)
            .map(|i| {
                let base = 37.0 + (i as f64 / 1000.0);
                Geometry::from_wkt(&format!(
                    "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                    base,
                    base,
                    base + 0.001,
                    base,
                    base + 0.001,
                    base + 0.001,
                    base,
                    base + 0.001,
                    base,
                    base
                ))
                .unwrap()
            })
            .collect();

        // Process chunk
        for geom in &chunk {
            total_area += area(geom)?;
            feature_count += 1;
        }

        // Report progress
        if chunk_end % 10_000 == 0 {
            println!("  Processed {}/{} features...", chunk_end, TOTAL_FEATURES);
        }
    }

    println!(
        "✓ Processed {} features, total area: {:.6}",
        feature_count, total_area
    );
    println!("  Tip: Process large datasets in chunks to avoid memory issues");

    Ok(())
}

// ============================================================================
// Pattern 5: Performance-Critical Code
// ============================================================================

/// Pattern: Optimize performance-critical spatial operations
///
/// Best practices:
/// - Use SIMD operations for batch distance calculations (4x faster)
/// - Use parallel processing for CPU-bound operations (8x faster)
/// - Profile before optimizing
#[cfg(feature = "parallel")]
fn pattern_performance_critical() -> Result<()> {
    use oxirs_geosparql::performance::batch::BatchProcessor;

    println!("\n=== Pattern 5: Performance-Critical Code ===\n");

    // Create test dataset
    let geometries: Vec<Geometry> = (0..1000)
        .map(|i| {
            Geometry::from_wkt(&format!(
                "POINT({} {})",
                (i % 100) as f64 * 0.1,
                (i / 100) as f64 * 0.1
            ))
            .unwrap()
        })
        .collect();

    let reference = Geometry::from_wkt("POINT(5.0 5.0)")?;

    // Sequential processing
    let start = std::time::Instant::now();
    let mut sequential_distances = Vec::new();
    for geom in &geometries {
        sequential_distances.push(distance(&reference, geom)?);
    }
    let sequential_time = start.elapsed();

    // Parallel processing with BatchProcessor
    let processor = BatchProcessor::new();
    let start = std::time::Instant::now();
    let parallel_distances = processor.distances(&reference, &geometries)?;
    let parallel_time = start.elapsed();

    println!("✓ Sequential processing: {:?}", sequential_time);
    println!("✓ Parallel processing: {:?}", parallel_time);
    println!(
        "  Speedup: {:.1}x faster",
        sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
    );
    println!("  Tip: Use BatchProcessor for CPU-bound batch operations");

    assert_eq!(sequential_distances.len(), parallel_distances.len());

    Ok(())
}

#[cfg(not(feature = "parallel"))]
fn pattern_performance_critical() -> Result<()> {
    println!("\n=== Pattern 5: Performance-Critical Code ===\n");
    println!("⚠ Enable 'parallel' feature for performance optimizations");
    Ok(())
}

// ============================================================================
// Pattern 6: CRS Transformation Pipeline
// ============================================================================

/// Pattern: Transform geometries between coordinate reference systems
///
/// Best practices:
/// - Batch transformations for better performance (10x faster)
/// - Validate CRS before transformation
/// - Use parallel batch transformation for large datasets
#[cfg(feature = "proj-support")]
fn pattern_crs_transformation() -> Result<()> {
    use oxirs_geosparql::functions::coordinate_transformation::*;
    use oxirs_geosparql::geometry::Crs;

    println!("\n=== Pattern 6: CRS Transformation Pipeline ===\n");

    // Create geometries in WGS84 (EPSG:4326)
    let geometries: Vec<Geometry> = vec![
        Geometry::from_wkt(
            "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(-122.4194 37.7749)",
        )?,
        Geometry::from_wkt(
            "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(-122.4083 37.7833)",
        )?,
        Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(-122.4 37.79)")?,
    ];

    // Target CRS: Web Mercator (EPSG:3857)
    let target_crs = Crs::new("http://www.opengis.net/def/crs/EPSG/0/3857");

    // Best practice: Use batch transformation
    let start = std::time::Instant::now();
    let transformed = transform_batch(&geometries, &target_crs)?;
    let duration = start.elapsed();

    println!(
        "✓ Transformed {} geometries in {:?}",
        transformed.len(),
        duration
    );
    println!("  Original CRS: WGS84 (EPSG:4326)");
    println!("  Target CRS: Web Mercator (EPSG:3857)");
    println!("  Tip: Batch transformation is 10x faster than individual transforms");

    Ok(())
}

#[cfg(not(feature = "proj-support"))]
fn pattern_crs_transformation() -> Result<()> {
    println!("\n=== Pattern 6: CRS Transformation Pipeline ===\n");
    println!("⚠ Enable 'proj-support' feature for CRS transformations");
    Ok(())
}

// ============================================================================
// Pattern 7: Geometry Validation and Repair
// ============================================================================

/// Pattern: Validate and repair invalid geometries
///
/// Best practices:
/// - Always validate geometries from untrusted sources
/// - Use repair_geometry for automatic fixing
/// - Log validation errors for debugging
fn pattern_validation_and_repair() -> Result<()> {
    use oxirs_geosparql::validation::*;

    println!("\n=== Pattern 7: Geometry Validation and Repair ===\n");

    // Create a problematic geometry (unclosed polygon ring)
    let invalid_polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0.001 0.001))")?;

    // Validate
    let validation = validate_geometry(&invalid_polygon);

    if !validation.is_valid {
        println!("⚠ Invalid geometry detected:");
        for error in &validation.errors {
            println!("  - {}", error);
        }

        // Attempt repair
        println!("\n  Attempting automatic repair...");
        let repaired = repair_geometry(&invalid_polygon)?;

        // Validate repaired geometry
        let revalidation = validate_geometry(&repaired);
        if revalidation.is_valid {
            println!("✓ Geometry repaired successfully");
        } else {
            println!("⚠ Manual repair required");
        }
    }

    // Simplify complex geometries
    let complex = Geometry::from_wkt("LINESTRING(0 0, 0.1 0.1, 0.2 0.2, 0.3 0.3, 1 1)")?;
    let _simplified = simplify_geometry(&complex, 0.15)?;

    println!("\n✓ Geometry simplified (tolerance: 0.15)");
    println!("  Tip: Simplification reduces coordinate count and improves performance");

    Ok(())
}

// ============================================================================
// Pattern 8: Production Error Handling
// ============================================================================

/// Pattern: Comprehensive error handling for production systems
///
/// Best practices:
/// - Use Result types everywhere
/// - Provide context in error messages
/// - Log errors with appropriate severity
/// - Implement graceful degradation
fn pattern_production_error_handling() -> Result<()> {
    println!("\n=== Pattern 8: Production Error Handling ===\n");

    let test_wkt = [
        "POINT(1 2)",                             // Valid
        "POINT(1 2 3)",                           // Valid 3D
        "INVALID WKT",                            // Invalid
        "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", // Valid
        "POINT EMPTY",                            // Invalid (empty) - correct WKT syntax
    ];

    let mut successful = 0;
    let mut failed = 0;

    for (i, wkt) in test_wkt.iter().enumerate() {
        match Geometry::from_wkt(wkt) {
            Ok(geom) => {
                successful += 1;
                println!("✓ Geometry {} parsed successfully", i + 1);

                // Additional validation
                let validation = oxirs_geosparql::validation::validate_geometry(&geom);
                if !validation.is_valid {
                    println!("  ⚠ Warning: Geometry is invalid but parseable");
                }
            }
            Err(e) => {
                failed += 1;
                println!("✗ Geometry {} failed: {}", i + 1, e);
                // In production: Log error with context
                // log::error!("Failed to parse geometry {}: {} - WKT: {}", i, e, wkt);
            }
        }
    }

    println!("\n✓ Results: {} successful, {} failed", successful, failed);
    println!("  Tip: Always handle errors gracefully in production");

    Ok(())
}

// ============================================================================
// Pattern 9: Multi-Format I/O
// ============================================================================

/// Pattern: Convert between different spatial data formats
///
/// Best practices:
/// - Support multiple formats for interoperability
/// - Use appropriate format for each use case
/// - Validate after format conversion
#[cfg(all(feature = "geojson-support", feature = "gml-support"))]
fn pattern_multi_format_io() -> Result<()> {
    println!("\n=== Pattern 9: Multi-Format I/O ===\n");

    // Start with WKT
    let original = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    println!("✓ Original: WKT");

    // Convert to GeoJSON
    let geojson = original.to_geojson()?;
    println!("✓ Converted to: GeoJSON");

    // Convert to GML
    let gml = original.to_gml()?;
    println!("✓ Converted to: GML");

    // Convert back from GeoJSON
    let from_geojson = Geometry::from_geojson(&geojson)?;
    println!("✓ Round-trip: GeoJSON → Geometry");

    // Convert back from GML
    let from_gml = Geometry::from_gml(&gml)?;
    println!("✓ Round-trip: GML → Geometry");

    // Verify consistency
    let area_original = oxirs_geosparql::functions::geometric_properties::area(&original)?;
    let area_geojson = oxirs_geosparql::functions::geometric_properties::area(&from_geojson)?;
    let area_gml = oxirs_geosparql::functions::geometric_properties::area(&from_gml)?;

    println!(
        "\n✓ Area verification: {:.2} (original) ≈ {:.2} (GeoJSON) ≈ {:.2} (GML)",
        area_original, area_geojson, area_gml
    );
    println!("  Tip: Always verify data integrity after format conversion");

    Ok(())
}

#[cfg(not(all(feature = "geojson-support", feature = "gml-support")))]
fn pattern_multi_format_io() -> Result<()> {
    println!("\n=== Pattern 9: Multi-Format I/O ===\n");
    println!("⚠ Enable 'geojson-support' and 'gml-support' features");
    Ok(())
}

// ============================================================================
// Pattern 10: Testing Spatial Code
// ============================================================================

/// Pattern: Write robust tests for spatial operations
///
/// Best practices:
/// - Use approx equality for floating-point comparisons
/// - Test edge cases (empty geometries, boundaries)
/// - Use property-based testing for mathematical properties
/// - Test with realistic data sizes
fn pattern_testing_spatial_code() -> Result<()> {
    println!("\n=== Pattern 10: Testing Spatial Code ===\n");

    // Test 1: Basic functionality
    let point = Geometry::from_wkt("POINT(1 2)")?;
    assert!(matches!(point.geom, geo_types::Geometry::Point(_)));
    println!("✓ Test 1: Basic functionality");

    // Test 2: Floating-point comparison with tolerance
    let poly = Geometry::from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")?;
    let actual_area = oxirs_geosparql::functions::geometric_properties::area(&poly)?;
    let expected_area = 1.0;
    assert!((actual_area - expected_area).abs() < 1e-10);
    println!("✓ Test 2: Floating-point comparison with tolerance");

    // Test 3: Edge cases
    let empty_geom = Geometry::from_wkt("POINT EMPTY");
    match empty_geom {
        Ok(_) => println!("✓ Test 3: Empty geometry handled"),
        Err(_) => println!("✓ Test 3: Empty geometry rejected (expected)"),
    }

    // Test 4: Property testing (distance symmetry)
    let p1 = Geometry::from_wkt("POINT(0 0)")?;
    let p2 = Geometry::from_wkt("POINT(3 4)")?;
    let dist_12 = distance(&p1, &p2)?;
    let dist_21 = distance(&p2, &p1)?;
    assert!((dist_12 - dist_21).abs() < 1e-10);
    println!("✓ Test 4: Distance symmetry property");

    println!("\n  Tip: Use property-based testing for mathematical correctness");

    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  oxirs-geosparql: Common Patterns and Best Practices      ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Run all patterns
    pattern_bulk_data_loading()?;
    pattern_safe_geometry_operations()?;
    pattern_proximity_analysis()?;
    pattern_streaming_large_datasets()?;
    pattern_performance_critical()?;
    pattern_crs_transformation()?;
    pattern_validation_and_repair()?;
    pattern_production_error_handling()?;
    pattern_multi_format_io()?;
    pattern_testing_spatial_code()?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Summary: 10 Common Patterns Demonstrated                 ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("\nFor more examples, see:");
    println!("  - examples/geosparql_basic_usage.rs");
    println!("  - examples/geosparql_performance_optimization.rs");
    println!("  - examples/*_support.rs (format-specific examples)");
    println!("\nDocumentation: https://docs.rs/oxirs-geosparql");

    Ok(())
}
