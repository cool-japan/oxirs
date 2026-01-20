//! Comparison of Pure Rust buffer vs GEOS backend
//!
//! This example demonstrates the hybrid buffer strategy where:
//! - Polygon/MultiPolygon → uses rust-buffer (pure Rust)
//! - Point/LineString → uses geos-backend (C++ library)
//!
//! Run with: cargo run --example buffer_comparison --features rust-buffer,geos-backend

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::geometric_operations::buffer;
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Buffer Backend Comparison ===\n");

    // Test 1: Polygon buffer - Uses Pure Rust
    println!("1. POLYGON BUFFER (Pure Rust Backend):");
    println!("   When you buffer a Polygon, oxirs-geosparql automatically uses");
    println!("   the pure Rust implementation (geo-buffer crate).\n");

    let square = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    println!("   Original: {}", square.to_wkt());

    let buffered_square = buffer(&square, 2.0)?;
    println!("   After buffer(2.0): {}", buffered_square.to_wkt());
    println!("   ✅ Used: Pure Rust (geo-buffer)\n");

    // Test 2: Point buffer - Uses GEOS
    #[cfg(feature = "geos-backend")]
    {
        println!("2. POINT BUFFER (GEOS Backend):");
        println!("   When you buffer a Point, oxirs-geosparql automatically uses");
        println!("   the GEOS backend (C++ library).\n");

        let point = Geometry::from_wkt("POINT(5 5)")?;
        println!("   Original: {}", point.to_wkt());

        let buffered_point = buffer(&point, 3.0)?;
        println!("   After buffer(3.0): {}", buffered_point.to_wkt());
        println!("   Result type: {}", buffered_point.geometry_type());
        println!("   ✅ Used: GEOS backend (creates circular polygon)\n");
    }

    #[cfg(not(feature = "geos-backend"))]
    {
        println!("2. POINT BUFFER (GEOS Backend):");
        println!("   ❌ GEOS backend not available - Point buffer requires geos-backend feature\n");
    }

    // Test 3: LineString buffer - Uses GEOS
    #[cfg(feature = "geos-backend")]
    {
        println!("3. LINESTRING BUFFER (GEOS Backend):");
        println!("   When you buffer a LineString, oxirs-geosparql automatically uses");
        println!("   the GEOS backend.\n");

        let line = Geometry::from_wkt("LINESTRING(0 0, 10 10, 20 5)")?;
        println!("   Original: {}", line.to_wkt());

        let buffered_line = buffer(&line, 2.0)?;
        println!("   After buffer(2.0): {}", buffered_line.to_wkt());
        println!("   Result type: {}", buffered_line.geometry_type());
        println!("   ✅ Used: GEOS backend (creates rounded corridor)\n");
    }

    #[cfg(not(feature = "geos-backend"))]
    {
        println!("3. LINESTRING BUFFER (GEOS Backend):");
        println!(
            "   ❌ GEOS backend not available - LineString buffer requires geos-backend feature\n"
        );
    }

    // Test 4: MultiPolygon buffer - Uses Pure Rust
    println!("4. MULTIPOLYGON BUFFER (Pure Rust Backend):");
    println!("   MultiPolygon also uses pure Rust implementation.\n");

    let multi = Geometry::from_wkt(
        "MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)), ((10 10, 15 10, 15 15, 10 15, 10 10)))",
    )?;
    println!("   Original: Two separate 5x5 squares");

    let buffered_multi = buffer(&multi, 1.0)?;
    println!("   After buffer(1.0): {}", buffered_multi.to_wkt());
    println!("   ✅ Used: Pure Rust (geo-buffer)\n");

    // Test 5: Performance comparison hint
    println!("\n=== BACKEND SELECTION STRATEGY ===\n");
    println!("The buffer() function automatically selects the optimal backend:\n");

    println!("📦 PURE RUST (rust-buffer):");
    println!("   ✅ Polygon, MultiPolygon");
    println!("   ✅ No C++ dependencies");
    println!("   ✅ Easy cross-compilation");
    println!("   ✅ Smaller binary size");
    println!("   ✅ Fast for polygon operations\n");

    println!("🔧 GEOS BACKEND (geos-backend):");
    println!("   ✅ Point, LineString, MultiPoint, MultiLineString");
    println!("   ✅ All geometry types supported");
    println!("   ✅ Industry-standard (PostGIS uses GEOS)");
    println!("   ✅ Advanced cap/join styles");
    println!("   ⚠️  Requires GEOS C++ library installation\n");

    println!("🎯 HYBRID STRATEGY:");
    println!("   The buffer() function provides the best of both worlds:");
    println!("   • Polygon/MultiPolygon → Pure Rust (when available)");
    println!("   • Other geometry types → GEOS backend (when available)");
    println!("   • Graceful fallback if features are missing\n");

    // Test 6: Feature flag demonstration
    println!("\n=== FEATURE FLAG CONFIGURATION ===\n");

    #[cfg(all(feature = "rust-buffer", feature = "geos-backend"))]
    {
        println!("Current configuration: ✅ Both backends available");
        println!("   • rust-buffer: ENABLED (Pure Rust for Polygon/MultiPolygon)");
        println!("   • geos-backend: ENABLED (GEOS for all geometry types)");
        println!("\n   This is the RECOMMENDED configuration for maximum compatibility.");
    }

    #[cfg(all(feature = "rust-buffer", not(feature = "geos-backend")))]
    {
        println!("Current configuration: ⚠️ Partial support");
        println!("   • rust-buffer: ENABLED (Pure Rust for Polygon/MultiPolygon)");
        println!("   • geos-backend: DISABLED");
        println!("\n   Point and LineString buffers will not work.");
        println!("   Consider enabling geos-backend for full support.");
    }

    #[cfg(all(not(feature = "rust-buffer"), feature = "geos-backend"))]
    {
        println!("Current configuration: ✅ GEOS-only");
        println!("   • rust-buffer: DISABLED");
        println!("   • geos-backend: ENABLED (GEOS for all geometry types)");
        println!("\n   All geometry types work, but requires GEOS C++ library.");
    }

    #[cfg(not(any(feature = "rust-buffer", feature = "geos-backend")))]
    {
        println!("Current configuration: ❌ No buffer support");
        println!("   • rust-buffer: DISABLED");
        println!("   • geos-backend: DISABLED");
        println!("\n   Buffer operations will not work.");
        println!("   Enable at least one feature to use buffer().");
    }

    println!("\n=== Example completed successfully! ===");

    Ok(())
}
