//! Comparison of Pure Rust buffer vs GEOS backend
//!
//! This example demonstrates the buffer strategy where:
//! - Polygon/MultiPolygon → uses rust-buffer (pure Rust, this crate)
//! - Point/LineString → requires GEOS, provided by the quarantined
//!   `oxirs-geosparql-adapter-geos` crate (publish = false)
//!
//! Run with: cargo run --example buffer_comparison --features rust-buffer

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

    // Test 2: Point buffer - requires GEOS (quarantined into the adapter crate)
    println!("2. POINT BUFFER (GEOS adapter):");
    println!("   Point buffering requires GEOS, which is provided by the");
    println!("   `oxirs-geosparql-adapter-geos` crate (publish = false). Call");
    println!("   `oxirs_geosparql_adapter_geos::buffer(&point, 3.0)` there.\n");

    // Test 3: LineString buffer - requires GEOS (quarantined into the adapter crate)
    println!("3. LINESTRING BUFFER (GEOS adapter):");
    println!("   LineString buffering likewise requires the");
    println!("   `oxirs-geosparql-adapter-geos` crate.\n");

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

    #[cfg(feature = "rust-buffer")]
    {
        println!("Current configuration: Pure-Rust buffer ENABLED");
        println!("   • rust-buffer: ENABLED (Pure Rust for Polygon/MultiPolygon)");
        println!("   • GEOS (Point/LineString, cap/join styles): provided by the");
        println!("     oxirs-geosparql-adapter-geos crate (publish = false).");
    }

    #[cfg(not(feature = "rust-buffer"))]
    {
        println!("Current configuration: no Pure-Rust buffer feature enabled");
        println!("   • rust-buffer: DISABLED (enable it for Polygon/MultiPolygon buffering)");
        println!("   • GEOS buffering: use the oxirs-geosparql-adapter-geos crate.");
    }

    println!("\n=== Example completed successfully! ===");

    Ok(())
}
