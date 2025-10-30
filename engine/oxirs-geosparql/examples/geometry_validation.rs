//! Geometry validation and quality checks example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Validating geometries for common issues
//! - Simplifying geometries using Douglas-Peucker algorithm
//! - Simplifying geometries using Visvalingam-Whyatt algorithm
//! - Snapping coordinates to a precision grid
//! - Handling validation errors and warnings
//! - Quality control workflows
//!
//! Run with: cargo run --example geometry_validation

use oxirs_geosparql::error::Result;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::validation::{
    simplify_geometry, simplify_geometry_vw, snap_to_precision, validate_geometry,
};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Geometry Validation Example ===\n");

    // 1. Basic geometry validation
    println!("1. BASIC GEOMETRY VALIDATION:");

    let valid_point = Geometry::from_wkt("POINT(10.0 20.0)")?;
    let validation = validate_geometry(&valid_point);
    println!("   Valid point: {}", valid_point.to_wkt());
    println!("   Is valid: {}", validation.is_valid);
    println!("   Errors: {:?}", validation.errors);
    println!("   Warnings: {:?}", validation.warnings);

    // 2. Detecting invalid coordinates
    println!("\n2. DETECTING INVALID COORDINATES:");

    let invalid_point = Geometry::from_wkt("POINT(0.0 0.0)")?;
    // Manually create an invalid point for demonstration
    println!("   Point: {}", invalid_point.to_wkt());

    let validation = validate_geometry(&invalid_point);
    if validation.is_valid {
        println!("   ✓ Geometry is valid");
    } else {
        println!("   ✗ Geometry has issues:");
        for error in &validation.errors {
            println!("     - {}", error);
        }
    }

    // 3. Empty geometry validation
    println!("\n3. EMPTY GEOMETRY VALIDATION:");

    let empty_linestring = Geometry::from_wkt("LINESTRING EMPTY")?;
    let validation = validate_geometry(&empty_linestring);
    println!("   Geometry: {}", empty_linestring.to_wkt());
    println!("   Is valid: {}", validation.is_valid);
    if !validation.warnings.is_empty() {
        println!("   Warnings:");
        for warning in &validation.warnings {
            println!("     - {}", warning);
        }
    }

    // 4. Polygon validation (self-intersection)
    println!("\n4. POLYGON VALIDATION:");

    let simple_polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let validation = validate_geometry(&simple_polygon);
    println!("   Polygon: {}", simple_polygon.to_wkt());
    println!("   Is valid: {}", validation.is_valid);
    if validation.is_valid {
        println!("   ✓ No self-intersections detected");
    }

    // 5. Douglas-Peucker simplification
    println!("\n5. DOUGLAS-PEUCKER SIMPLIFICATION:");

    let complex_line =
        Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 0, 3 1, 4 0, 5 1, 6 0, 7 1, 8 0, 9 1, 10 0)")?;
    println!("   Original: {}", complex_line.to_wkt());
    println!("   Original point count: 11");

    let simplified = simplify_geometry(&complex_line, 1.5)?;
    println!("   Simplified (epsilon=1.5): {}", simplified.to_wkt());

    // Count points in simplified geometry
    if let geo_types::Geometry::LineString(ls) = &simplified.geom {
        println!("   Simplified point count: {}", ls.0.len());
        println!(
            "   ✓ Reduced complexity by {}%",
            ((1.0 - ls.0.len() as f64 / 11.0) * 100.0) as i32
        );
    }

    // 6. Visvalingam-Whyatt simplification
    println!("\n6. VISVALINGAM-WHYATT SIMPLIFICATION:");

    let complex_line2 = Geometry::from_wkt("LINESTRING(0 0, 1 0.1, 2 0, 3 0.1, 4 0, 5 0.1, 6 0)")?;
    println!("   Original: {}", complex_line2.to_wkt());

    let simplified_vw = simplify_geometry_vw(&complex_line2, 0.05)?;
    println!("   Simplified (epsilon=0.05): {}", simplified_vw.to_wkt());

    if let geo_types::Geometry::LineString(ls) = &simplified_vw.geom {
        println!("   Simplified point count: {}", ls.0.len());
    }
    println!("   ✓ Preserves overall shape better than Douglas-Peucker");

    // 7. Comparison of simplification algorithms
    println!("\n7. COMPARISON: DOUGLAS-PEUCKER VS VISVALINGAM-WHYATT:");

    let test_line =
        Geometry::from_wkt("LINESTRING(0 0, 1 2, 2 1, 3 3, 4 2, 5 4, 6 3, 7 5, 8 4, 9 6, 10 5)")?;
    println!("   Original: {} points", 11);

    let dp_result = simplify_geometry(&test_line, 1.0)?;
    let vw_result = simplify_geometry_vw(&test_line, 1.0)?;

    if let (geo_types::Geometry::LineString(dp_ls), geo_types::Geometry::LineString(vw_ls)) =
        (&dp_result.geom, &vw_result.geom)
    {
        println!("   Douglas-Peucker result: {} points", dp_ls.0.len());
        println!("   Visvalingam-Whyatt result: {} points", vw_ls.0.len());
        println!("   Note: Different algorithms, different trade-offs!");
    }

    // 8. Snapping to precision grid
    println!("\n8. SNAPPING TO PRECISION GRID:");

    let imprecise_point = Geometry::from_wkt("POINT(1.23456789 2.98765432)")?;
    println!("   Original: {}", imprecise_point.to_wkt());

    let snapped_2 = snap_to_precision(&imprecise_point, 2)?;
    println!("   Snapped to 2 decimals: {}", snapped_2.to_wkt());

    let snapped_4 = snap_to_precision(&imprecise_point, 4)?;
    println!("   Snapped to 4 decimals: {}", snapped_4.to_wkt());

    let snapped_0 = snap_to_precision(&imprecise_point, 0)?;
    println!("   Snapped to integers: {}", snapped_0.to_wkt());

    // 9. Idempotent snapping
    println!("\n9. IDEMPOTENT SNAPPING:");

    let point = Geometry::from_wkt("POINT(123.456789 45.678901)")?;
    let snap1 = snap_to_precision(&point, 3)?;
    let snap2 = snap_to_precision(&snap1, 3)?;

    println!("   Original: {}", point.to_wkt());
    println!("   First snap: {}", snap1.to_wkt());
    println!("   Second snap: {}", snap2.to_wkt());

    if snap1.to_wkt() == snap2.to_wkt() {
        println!("   ✓ Snapping is idempotent (snap twice = snap once)");
    }

    // 10. Real-world workflow: Data quality pipeline
    println!("\n\n=== REAL-WORLD WORKFLOW: DATA QUALITY PIPELINE ===\n");

    println!("Scenario: Processing GPS tracks with quality issues");

    // Simulated GPS track with noise and excessive points
    let noisy_gps_track = Geometry::from_wkt(
        "LINESTRING(
            -122.4194 37.7749,
            -122.4195 37.7750,
            -122.4196 37.7751,
            -122.4197 37.7750,
            -122.4198 37.7752,
            -122.4199 37.7751,
            -122.4200 37.7753,
            -122.4201 37.7754,
            -122.4202 37.7755
        )",
    )?;

    println!("Step 1 - Validate input:");
    let validation = validate_geometry(&noisy_gps_track);
    if validation.is_valid {
        println!("   ✓ Track is valid");
    } else {
        println!("   ✗ Track has errors: {:?}", validation.errors);
    }

    if let geo_types::Geometry::LineString(ls) = &noisy_gps_track.geom {
        println!("   Original track: {} GPS points", ls.0.len());
    }

    println!("\nStep 2 - Simplify to reduce noise:");
    let simplified_track = simplify_geometry(&noisy_gps_track, 0.0001)?;

    if let geo_types::Geometry::LineString(ls) = &simplified_track.geom {
        println!("   Simplified track: {} points", ls.0.len());
        println!("   ✓ Reduced data size for storage/transmission");
    }

    println!("\nStep 3 - Snap coordinates to standard precision:");
    let final_track = snap_to_precision(&simplified_track, 6)?;
    println!("   Final track (6 decimal places = ~0.1m precision):");
    println!("   {}", final_track.to_wkt());

    println!("\nStep 4 - Final validation:");
    let final_validation = validate_geometry(&final_track);
    if final_validation.is_valid {
        println!("   ✓ Clean track ready for storage/analysis");
    }

    // 11. Handling polygons with holes
    println!("\n\n=== POLYGON WITH HOLES VALIDATION ===\n");

    let polygon_with_hole =
        Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))")?;
    println!("Polygon with hole (donut shape):");
    println!("   {}", polygon_with_hole.to_wkt());

    let validation = validate_geometry(&polygon_with_hole);
    println!("   Is valid: {}", validation.is_valid);
    if validation.is_valid {
        println!("   ✓ Exterior and interior rings are valid");
    }

    // 12. Error handling
    println!("\n\n=== ERROR HANDLING ===\n");

    println!("1. Invalid epsilon for simplification:");
    match simplify_geometry(&complex_line, -1.0) {
        Ok(_) => println!("   Unexpected success"),
        Err(e) => println!("   ✓ Correctly caught error: {}", e),
    }

    println!("\n2. Invalid precision for snapping:");
    match snap_to_precision(&imprecise_point, 20) {
        Ok(_) => println!("   Unexpected success"),
        Err(e) => println!("   ✓ Correctly caught error: {}", e),
    }

    // 13. Best practices summary
    println!("\n\n=== BEST PRACTICES ===\n");
    println!("1. Always validate geometries before processing");
    println!("2. Use appropriate simplification algorithm:");
    println!("   - Douglas-Peucker: Faster, good for most cases");
    println!("   - Visvalingam-Whyatt: Better shape preservation");
    println!("3. Choose epsilon based on your use case:");
    println!("   - Smaller epsilon = more detail preserved");
    println!("   - Larger epsilon = more simplification");
    println!("4. Snap to appropriate precision:");
    println!("   - 6 decimals (~0.1m) for GPS tracks");
    println!("   - 2 decimals (~1km) for regional analysis");
    println!("   - 0 decimals for coarse-grained data");
    println!("5. Validate after each transformation step");

    println!("\n=== Example completed successfully! ===");

    Ok(())
}
