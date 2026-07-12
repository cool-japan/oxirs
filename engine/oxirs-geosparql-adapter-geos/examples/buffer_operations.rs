//! GEOS buffer operations example (relocated from oxirs-geosparql under the
//! Pure Rust Policy v2 quarantine of the GEOS C FFI).
//!
//! Demonstrates positive/negative buffers, cap/join styles, and boundary
//! extraction — capabilities the Pure-Rust parent only provides for
//! Polygon/MultiPolygon (via `geo-buffer`); GEOS covers every geometry type.
//!
//! Run with: cargo run -p oxirs-geosparql-adapter-geos --example buffer_operations
//! (requires the GEOS C library: macOS `brew install geos`, Ubuntu
//! `apt-get install libgeos-dev`, Fedora `dnf install geos-devel`).

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::geometric_operations::{BufferParams, CapStyle, JoinStyle};
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql_adapter_geos::{boundary, buffer, buffer_with_params};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Buffer Operations Example (GEOS adapter) ===\n");

    // 1. Basic positive buffer (expansion)
    println!("1. BASIC POSITIVE BUFFER (Expansion):");
    let point = Geometry::from_wkt("POINT(0 0)")?;
    let buffered = buffer(&point, 1.0)?;
    println!("   POINT(0 0) buffered 1.0 -> {}", buffered.geometry_type());

    // 2. Negative buffer (erosion)
    println!("2. NEGATIVE BUFFER (Erosion/Inset):");
    let large_poly = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))")?;
    let shrunk = buffer(&large_poly, -10.0)?;
    println!("   100x100 square buffered -10.0 -> {}", shrunk.to_wkt());

    // 3. Flat cap style
    println!("3. FLAT CAP STYLE:");
    let line = Geometry::from_wkt("LINESTRING(0 0, 10 0)")?;
    let flat_params = BufferParams {
        cap_style: CapStyle::Flat,
        ..Default::default()
    };
    let flat_buffer = buffer_with_params(&line, 2.0, &flat_params)?;
    println!(
        "   flat-capped line buffer -> {}",
        flat_buffer.geometry_type()
    );

    // 4. Mitre join style
    println!("4. MITRE JOIN STYLE:");
    let angular_line = Geometry::from_wkt("LINESTRING(0 0, 5 0, 5 5)")?;
    let mitre_params = BufferParams {
        join_style: JoinStyle::Mitre,
        mitre_limit: 5.0,
        ..Default::default()
    };
    let mitre_buffer = buffer_with_params(&angular_line, 1.0, &mitre_params)?;
    println!("   mitre-joined buffer -> {}", mitre_buffer.geometry_type());

    // 5. Square cap style
    println!("5. SQUARE CAP STYLE:");
    let square_params = BufferParams {
        cap_style: CapStyle::Square,
        ..Default::default()
    };
    let square_buffer = buffer_with_params(&line, 2.0, &square_params)?;
    println!(
        "   square-capped line buffer -> {}",
        square_buffer.geometry_type()
    );

    // 6. Boundary operations
    println!("\n=== BOUNDARY OPERATIONS ===");
    let boundary_line = Geometry::from_wkt("LINESTRING(0 0, 10 0, 10 10)")?;
    println!(
        "6. Boundary of LINESTRING -> {}",
        boundary(&boundary_line)?.to_wkt()
    );

    let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    println!("7. Boundary of POLYGON -> {}", boundary(&polygon)?.to_wkt());

    let donut = Geometry::from_wkt(
        "POLYGON((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))",
    )?;
    println!(
        "8. Boundary of POLYGON-with-hole -> {}",
        boundary(&donut)?.to_wkt()
    );

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
