//! Buffer operations example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Positive buffers (expansion) for creating zones
//! - Negative buffers (erosion/inset) for shrinking geometries
//! - Different cap styles (round, flat, square)
//! - Different join styles (round, mitre, bevel)
//! - Boundary operations for extracting geometry boundaries
//!
//! Run with: cargo run --example buffer_operations --features geos-backend

use oxirs_geosparql::error::Result;

#[cfg(feature = "geos-backend")]
use oxirs_geosparql::functions::geometric_operations::{
    boundary, buffer, buffer_with_params, BufferParams, CapStyle, JoinStyle,
};
#[cfg(feature = "geos-backend")]
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Buffer Operations Example ===\n");

    #[cfg(feature = "geos-backend")]
    {
        // 1. Basic positive buffer (expansion)
        println!("1. BASIC POSITIVE BUFFER (Expansion):");
        let point = Geometry::from_wkt("POINT(0 0)")?;
        println!("   Original: {}", point.to_wkt());

        let buffered = buffer(&point, 1.0)?;
        println!("   After 1.0 buffer: {}", buffered.to_wkt());
        println!("   Result type: {}", buffered.geometry_type());
        println!("   Use case: Create protection zones, service areas\n");

        // 2. Negative buffer (erosion)
        println!("2. NEGATIVE BUFFER (Erosion/Inset):");
        let large_poly = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))")?;
        println!("   Original: Square 100x100");

        let shrunk = buffer(&large_poly, -10.0)?;
        println!("   After -10.0 buffer: {}", shrunk.to_wkt());
        println!("   Use case: Create setbacks, inner boundaries\n");

        // 3. Buffer with round cap style (default)
        println!("3. ROUND CAP STYLE (Default):");
        let line = Geometry::from_wkt("LINESTRING(0 0, 10 0)")?;
        println!("   Original line: {}", line.to_wkt());

        let round_buffer = buffer(&line, 2.0)?;
        println!("   With round caps: {}", round_buffer.to_wkt());
        println!("   Creates smooth rounded ends\n");

        // 4. Buffer with flat cap style
        println!("4. FLAT CAP STYLE:");
        let flat_params = BufferParams {
            cap_style: CapStyle::Flat,
            ..Default::default()
        };

        let flat_buffer = buffer_with_params(&line, 2.0, &flat_params)?;
        println!("   With flat caps: {}", flat_buffer.to_wkt());
        println!("   Creates rectangular ends without extension\n");

        // 5. Buffer with square cap style
        println!("5. SQUARE CAP STYLE:");
        let square_params = BufferParams {
            cap_style: CapStyle::Square,
            ..Default::default()
        };

        let square_buffer = buffer_with_params(&line, 2.0, &square_params)?;
        println!("   With square caps: {}", square_buffer.to_wkt());
        println!("   Extends beyond endpoints by buffer distance\n");

        // 6. Buffer with mitre join style
        println!("6. MITRE JOIN STYLE:");
        let angular_line = Geometry::from_wkt("LINESTRING(0 0, 5 0, 5 5)")?;
        let mitre_params = BufferParams {
            join_style: JoinStyle::Mitre,
            mitre_limit: 5.0,
            ..Default::default()
        };

        let mitre_buffer = buffer_with_params(&angular_line, 1.0, &mitre_params)?;
        println!("   Original: L-shaped line");
        println!("   With mitre joins: {}", mitre_buffer.to_wkt());
        println!("   Creates sharp corners at angles\n");

        // 7. Buffer with bevel join style
        println!("7. BEVEL JOIN STYLE:");
        let bevel_params = BufferParams {
            join_style: JoinStyle::Bevel,
            ..Default::default()
        };

        let bevel_buffer = buffer_with_params(&angular_line, 1.0, &bevel_params)?;
        println!("   With bevel joins: {}", bevel_buffer.to_wkt());
        println!("   Creates chamfered corners\n");

        // 8. High-quality buffer with more segments
        println!("8. HIGH-QUALITY BUFFER (More Segments):");
        let high_quality_params = BufferParams {
            quadrant_segments: 16, // More segments = smoother curves
            ..Default::default()
        };

        let smooth_buffer = buffer_with_params(&point, 5.0, &high_quality_params)?;
        println!("   16 segments per quadrant (64 total vertices)");
        println!("   Result: {}", smooth_buffer.to_wkt());
        println!("   Use case: High-precision mapping\n");

        // 9. Boundary operations
        println!("\n=== BOUNDARY OPERATIONS ===\n");

        println!("9. BOUNDARY OF LINESTRING:");
        let boundary_line = Geometry::from_wkt("LINESTRING(0 0, 10 0, 10 10)")?;
        let line_boundary = boundary(&boundary_line)?;
        println!("   Original: {}", boundary_line.to_wkt());
        println!("   Boundary: {}", line_boundary.to_wkt());
        println!("   (The two endpoints)\n");

        println!("10. BOUNDARY OF POLYGON:");
        let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
        let poly_boundary = boundary(&polygon)?;
        println!("   Original: Square polygon");
        println!("   Boundary: {}", poly_boundary.to_wkt());
        println!("   (The exterior ring)\n");

        println!("11. BOUNDARY OF POLYGON WITH HOLE:");
        let donut = Geometry::from_wkt(
            "POLYGON((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))",
        )?;
        let donut_boundary = boundary(&donut)?;
        println!("   Original: Polygon with hole (donut)");
        println!("   Boundary: {}", donut_boundary.to_wkt());
        println!("   (Both exterior and interior rings)\n");

        // 12. Real-world example: Urban planning
        println!("\n=== REAL-WORLD EXAMPLE: URBAN PLANNING ===\n");

        let building = Geometry::from_wkt("POLYGON((10 10, 30 10, 30 30, 10 30, 10 10))")?;
        println!("Building footprint: {}", building.to_wkt());

        // Fire safety zone
        let fire_zone = buffer(&building, 5.0)?;
        println!("\nFire safety zone (5m buffer):");
        println!("   {}", fire_zone.to_wkt());
        println!("   Use: Minimum distance for adjacent buildings");

        // Setback for landscaping
        let setback = buffer(&building, -2.0)?;
        println!("\nLandscaping setback (2m inset):");
        println!("   {}", setback.to_wkt());
        println!("   Use: Inner boundary for garden planning");

        // Noise barrier zone
        let road = Geometry::from_wkt("LINESTRING(0 0, 100 0)")?;
        let noise_params = BufferParams {
            cap_style: CapStyle::Flat,
            ..Default::default()
        };
        let noise_zone = buffer_with_params(&road, 25.0, &noise_params)?;
        println!("\nNoise barrier zone (25m from road with flat caps):");
        println!("   Road: {}", road.to_wkt());
        println!("   Noise zone: {}", noise_zone.to_wkt());
        println!("   Use: Determine affected residential areas");

        // 13. Multiple buffer rings
        println!("\n\n=== MULTIPLE BUFFER RINGS ===\n");

        let facility = Geometry::from_wkt("POINT(50 50)")?;
        println!("Facility location: {}", facility.to_wkt());

        for (distance, name) in [(5.0, "Immediate"), (10.0, "Near"), (20.0, "Extended")] {
            let ring = buffer(&facility, distance)?;
            println!("\n{} zone ({}m radius):", name, distance);
            println!("   Type: {}", ring.geometry_type());
        }

        // 14. Complex geometry buffering
        println!("\n\n=== COMPLEX GEOMETRY BUFFERING ===\n");

        let multi_line =
            Geometry::from_wkt("MULTILINESTRING((0 0, 10 0), (20 0, 30 0), (15 5, 15 15))")?;
        println!("Road network: {}", multi_line.to_wkt());

        let corridor = buffer(&multi_line, 3.0)?;
        println!("\nTransport corridor (3m buffer):");
        println!("   {}", corridor.to_wkt());
        println!("   Use: Right-of-way planning");

        println!("\n=== Example completed successfully! ===");
        println!("\nNote: This example requires the 'geos-backend' feature.");
        println!("Run with: cargo run --example buffer_operations --features geos-backend");
    }

    #[cfg(not(feature = "geos-backend"))]
    {
        println!("❌ This example requires the 'geos-backend' feature to be enabled.\n");
        println!("Please run with:");
        println!("   cargo run --example buffer_operations --features geos-backend\n");
        println!("Note: This requires GEOS library to be installed on your system:");
        println!("   macOS:   brew install geos");
        println!("   Ubuntu:  sudo apt-get install libgeos-dev");
        println!("   Windows: See https://trac.osgeo.org/geos/");
    }

    Ok(())
}
