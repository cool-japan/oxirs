//! Pure Rust buffer operations example for oxirs-geosparql
//!
//! This example demonstrates buffer operations using pure Rust implementation
//! (without requiring C++ GEOS library). The pure Rust buffer uses the
//! straight skeleton algorithm from the geo-buffer crate.
//!
//! Supported:
//! - Polygon and MultiPolygon geometries
//! - Positive buffers (expansion) and negative buffers (erosion)
//! - Polygons with holes
//!
//! Run with: cargo run --example pure_rust_buffer --features rust-buffer

use oxirs_geosparql::error::Result;

#[cfg(feature = "rust-buffer")]
use oxirs_geosparql::functions::geometric_operations::{buffer, buffer_rust};
#[cfg(feature = "rust-buffer")]
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Pure Rust Buffer Example ===\n");

    #[cfg(feature = "rust-buffer")]
    {
        // 1. Basic polygon buffer
        println!("1. BASIC POLYGON BUFFER:");
        let square = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
        println!("   Original square (10x10): {}", square.to_wkt());

        let expanded = buffer_rust(&square, 2.0)?;
        println!("   After +2.0 buffer: {}", expanded.to_wkt());
        println!("   Result type: {}", expanded.geometry_type());
        println!("   ✅ Pure Rust - No C++ dependencies!\n");

        // 2. Negative buffer (erosion)
        println!("2. NEGATIVE BUFFER (Erosion):");
        let large_square = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))")?;
        println!("   Original: 100x100 square");

        let shrunk = buffer_rust(&large_square, -10.0)?;
        println!("   After -10.0 buffer: {}", shrunk.to_wkt());
        println!("   Creates 10-unit inset from all sides\n");

        // 3. Polygon with hole (donut shape)
        println!("3. POLYGON WITH HOLE:");
        let donut = Geometry::from_wkt(
            "POLYGON((0 0, 50 0, 50 50, 0 50, 0 0), (10 10, 40 10, 40 40, 10 40, 10 10))",
        )?;
        println!("   Original donut: {}", donut.to_wkt());

        let buffered_donut = buffer_rust(&donut, 5.0)?;
        println!("   After +5.0 buffer: {}", buffered_donut.to_wkt());
        println!("   Handles holes correctly!\n");

        // 4. MultiPolygon
        println!("4. MULTIPOLYGON BUFFER:");
        let multi = Geometry::from_wkt(
            "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)), ((20 20, 30 20, 30 30, 20 30, 20 20)))",
        )?;
        println!("   Two separate squares");

        let multi_buffered = buffer_rust(&multi, 3.0)?;
        println!("   After +3.0 buffer: {}", multi_buffered.to_wkt());
        println!("   Each polygon buffered independently\n");

        // 5. Non-convex polygon
        println!("5. NON-CONVEX POLYGON (L-shape):");
        let l_shape = Geometry::from_wkt("POLYGON((0 0, 20 0, 20 10, 10 10, 10 20, 0 20, 0 0))")?;
        println!("   Original L-shape: {}", l_shape.to_wkt());

        let buffered_l = buffer_rust(&l_shape, 2.0)?;
        println!("   After +2.0 buffer: {}", buffered_l.to_wkt());
        println!("   Straight skeleton handles complex shapes!\n");

        // 6. Hybrid buffer (uses rust-buffer for Polygon)
        println!("6. HYBRID BUFFER (Automatic Backend Selection):");
        println!("   The buffer() function automatically chooses the best backend:");
        println!("   - Polygon/MultiPolygon → Pure Rust (geo-buffer)");
        println!("   - Point/LineString → GEOS backend (if available)\n");

        let test_poly = Geometry::from_wkt("POLYGON((0 0, 15 0, 15 15, 0 15, 0 0))")?;
        let hybrid_result = buffer(&test_poly, 3.0)?;
        println!("   Polygon buffer: {}", hybrid_result.to_wkt());
        println!("   ✅ Used Pure Rust implementation\n");

        // 7. Real-world example: Urban planning
        println!("7. REAL-WORLD EXAMPLE: URBAN PLANNING\n");

        let park = Geometry::from_wkt("POLYGON((100 100, 200 100, 200 200, 100 200, 100 100))")?;
        println!("   Park area: 100x100m square");

        // Create 20m buffer zone around park
        let buffer_zone = buffer_rust(&park, 20.0)?;
        println!("\n   20m buffer zone for noise protection:");
        println!("   {}", buffer_zone.to_wkt());

        // Create 10m setback inside park
        let usable_area = buffer_rust(&park, -10.0)?;
        println!("\n   10m setback (usable area inside park):");
        println!("   {}", usable_area.to_wkt());

        // 8. Complex urban zone with exclusion
        println!("\n8. COMPLEX URBAN ZONE:");
        let zone_with_exclusion = Geometry::from_wkt(
            "POLYGON((0 0, 300 0, 300 300, 0 300, 0 0), (100 100, 200 100, 200 200, 100 200, 100 100))"
        )?;
        println!("   Zone: 300x300m with 100x100m exclusion");

        let development_zone = buffer_rust(&zone_with_exclusion, -20.0)?;
        println!("   After 20m setback: {}", development_zone.to_wkt());
        println!("   Creates buildable area with proper clearances\n");

        // 9. Comparison with WKT round-trip
        println!("9. WKT ROUND-TRIP:");
        let original = Geometry::from_wkt("POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))")?;
        let buffered = buffer_rust(&original, 5.0)?;
        let wkt_output = buffered.to_wkt();
        let reparsed = Geometry::from_wkt(&wkt_output)?;

        println!("   Original → Buffer → WKT → Parse: ✅");
        println!("   Result type: {}", reparsed.geometry_type());

        println!("\n=== Pure Rust Buffer Advantages ===");
        println!("✅ No C++ dependencies (GEOS not required)");
        println!("✅ Easy cross-compilation");
        println!("✅ Smaller binary size");
        println!("✅ Handles Polygon/MultiPolygon perfectly");
        println!("✅ Supports holes and complex shapes");
        println!("✅ Based on straight skeleton algorithm");

        println!("\n=== Limitations ===");
        println!("⚠️  Only supports Polygon and MultiPolygon");
        println!("⚠️  Point and LineString require GEOS backend");
        println!("⚠️  Different algorithm than GEOS (may have slight differences)");

        println!("\n=== Example completed successfully! ===");
    }

    #[cfg(not(feature = "rust-buffer"))]
    {
        println!("❌ This example requires the 'rust-buffer' feature to be enabled.\n");
        println!("Please run with:");
        println!("   cargo run --example pure_rust_buffer --features rust-buffer\n");
        println!("Advantages of rust-buffer:");
        println!("  ✅ No C++ dependencies");
        println!("  ✅ Works for Polygon/MultiPolygon");
        println!("  ✅ Pure Rust implementation");
    }

    Ok(())
}
