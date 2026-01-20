//! Profiling Demo: Identify Performance Bottlenecks
//!
//! This example demonstrates how to use the profiling utilities
//! to measure and optimize spatial operations performance.

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::geometric_operations::*;
use oxirs_geosparql::functions::geometric_properties::*;
use oxirs_geosparql::functions::simple_features::*;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::index::SpatialIndex;
use oxirs_geosparql::performance::profiling::{ProfileScope, Profiler};
use oxirs_geosparql::profile_scope;

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║           Performance Profiling Demonstration             ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Create a profiler instance
    let mut profiler = Profiler::new();

    // Example 1: Profile WKT parsing
    demo_parsing_profiling(&mut profiler)?;

    // Example 2: Profile spatial operations
    demo_operations_profiling(&mut profiler)?;

    // Example 3: Profile spatial indexing
    demo_indexing_profiling(&mut profiler)?;

    // Example 4: Profile with RAII scope
    demo_scope_profiling(&mut profiler)?;

    // Print comprehensive performance report
    println!("\n");
    profiler.print_report();

    // Export to JSON
    {
        let json = profiler.export_json();
        println!("\n📊 JSON Export:");
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    }

    // Performance recommendations
    print_recommendations(&profiler);

    Ok(())
}

/// Demonstrate profiling WKT parsing performance
fn demo_parsing_profiling(profiler: &mut Profiler) -> Result<()> {
    println!("Example 1: Profiling WKT Parsing\n");

    let wkt_examples = vec![
        "POINT(1 2)",
        "LINESTRING(0 0, 1 1, 2 0)",
        "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
        "MULTIPOINT((0 0), (1 1), (2 2))",
    ];

    // Profile parsing of different geometry types
    for wkt in &wkt_examples {
        profiler.start("wkt_parsing");
        let _geom = Geometry::from_wkt(wkt)?;
        profiler.stop("wkt_parsing");
    }

    // Validate geometries (separate profiling)
    for wkt in &wkt_examples {
        let geom = Geometry::from_wkt(wkt)?;
        profiler.start("validation");
        let _validation = oxirs_geosparql::validation::validate_geometry(&geom);
        profiler.stop("validation");
    }

    println!("✓ Parsed and validated {} geometries\n", wkt_examples.len());

    Ok(())
}

/// Demonstrate profiling spatial operations
fn demo_operations_profiling(profiler: &mut Profiler) -> Result<()> {
    println!("Example 2: Profiling Spatial Operations\n");

    let poly1 = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let poly2 = Geometry::from_wkt("POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))")?;

    // Profile different operations
    profiler.start("area_calc");
    let _area1 = area(&poly1)?;
    let _area2 = area(&poly2)?;
    profiler.stop("area_calc");

    profiler.start("intersection");
    let _intersect = intersection(&poly1, &poly2)?;
    profiler.stop("intersection");

    profiler.start("union");
    let _union_geom = union(&poly1, &poly2)?;
    profiler.stop("union");

    profiler.start("difference");
    let _diff = difference(&poly1, &poly2)?;
    profiler.stop("difference");

    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    {
        profiler.start("buffer");
        let _buffer_geom = buffer(&poly1, 1.0)?;
        profiler.stop("buffer");
    }

    profiler.start("contains");
    let _contains = sf_contains(&poly1, &poly2)?;
    profiler.stop("contains");

    println!("✓ Profiled 6 spatial operations\n");

    Ok(())
}

/// Demonstrate profiling spatial indexing
fn demo_indexing_profiling(profiler: &mut Profiler) -> Result<()> {
    println!("Example 3: Profiling Spatial Indexing\n");

    // Generate test dataset
    profiler.start("data_generation");
    let geometries: Vec<Geometry> = (0..10_000)
        .map(|i| {
            let lat = 37.7 + (i as f64 / 1000.0) * 0.1;
            let lon = -122.4 + (i as f64 / 1000.0) * 0.1;
            Geometry::from_wkt(&format!("POINT({} {})", lon, lat)).unwrap()
        })
        .collect();
    profiler.stop("data_generation");

    // Profile bulk loading
    profiler.start("bulk_load");
    let index = SpatialIndex::bulk_load(geometries)?;
    profiler.stop("bulk_load");

    // Profile queries
    for _ in 0..10 {
        profiler.start("bbox_query");
        let _results = index.query_bbox(-122.4, 37.7, -122.3, 37.8);
        profiler.stop("bbox_query");

        profiler.start("nearest_query");
        let _nearest = index.nearest(-122.35, 37.75);
        profiler.stop("nearest_query");
    }

    println!("✓ Indexed 10,000 points and ran 20 queries\n");

    Ok(())
}

/// Demonstrate RAII-style profiling with ProfileScope
fn demo_scope_profiling(profiler: &mut Profiler) -> Result<()> {
    println!("Example 4: RAII-Style Profiling (Auto-Stop)\n");

    // Method 1: Using ProfileScope directly
    {
        let _scope = ProfileScope::new(profiler, "complex_computation");

        // Simulate expensive work
        let geom = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))")?;
        let _area = area(&geom)?;
        let _perimeter = length(&geom)?;

        // ProfileScope automatically stops profiling when it goes out of scope
    }

    // Method 2: Using profile_scope! macro
    profile_scope!(profiler, "batch_processing", {
        for i in 0..100 {
            let point = Geometry::from_wkt(&format!("POINT({} {})", i, i))?;
            let _validation = oxirs_geosparql::validation::validate_geometry(&point);
        }
    });

    println!("✓ Demonstrated automatic profiling scope management\n");

    Ok(())
}

/// Print performance recommendations based on profiling data
fn print_recommendations(profiler: &Profiler) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                  Performance Recommendations               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Analyze bulk_load performance
    if let Some(stats) = profiler.get_stats("bulk_load") {
        let ms = stats.total.as_secs_f64() * 1000.0;
        println!("📊 Bulk Load Performance:");
        println!("   - Time: {:.2}ms for 10,000 geometries", ms);
        if ms > 100.0 {
            println!("   ⚠ Consider using parallel bulk loading for larger datasets");
        } else {
            println!("   ✓ Performance is excellent");
        }
        println!();
    }

    // Analyze query performance
    if let Some(stats) = profiler.get_stats("bbox_query") {
        let avg_us = stats.average.as_micros();
        println!("📊 Query Performance:");
        println!("   - Average bbox query: {}µs", avg_us);
        if avg_us > 1000 {
            println!("   ⚠ Consider optimizing R-tree parameters");
        } else {
            println!("   ✓ Query performance is optimal");
        }
        println!();
    }

    // Analyze operation costs
    let expensive_ops = vec![("intersection", 1000), ("union", 1000), ("buffer", 1000)];

    for (op, threshold_us) in expensive_ops {
        if let Some(stats) = profiler.get_stats(op) {
            let avg_us = stats.average.as_micros();
            if avg_us > threshold_us {
                println!("⚠ Expensive operation detected: {}", op);
                println!("   - Average time: {}µs", avg_us);
                println!("   - Consider batching or caching results");
                println!();
            }
        }
    }

    println!("💡 General Tips:");
    println!("   - Use spatial indexes for proximity queries");
    println!("   - Batch operations when possible");
    println!("   - Profile before optimizing (measure, don't guess)");
    println!("   - Consider parallel processing for datasets >50k elements");
    println!("   - Cache expensive computations");
}
