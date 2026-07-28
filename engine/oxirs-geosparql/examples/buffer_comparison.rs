//! Buffer capabilities across geometry types.
//!
//! Buffering is Pure Rust and unconditional — it runs on `geo::algorithm::buffer`
//! (i_overlay-backed). This example used to contrast a Polygon-only straight-skeleton
//! path against a GEOS C-library backend for everything else; there is one path now.
//!
//! Run with: cargo run --example buffer_comparison

use oxirs_geosparql::functions::geometric_operations::{
    buffer, buffer_with_params, BufferParams, CapStyle, JoinStyle,
};
use oxirs_geosparql::geometry::Geometry;

fn main() {
    println!("=== OxiRS GeoSPARQL Buffer ===\n");

    println!("Every geometry type buffers, with the full OGC cap/join styles,");
    println!("with no C library involved.\n");

    // ---------------------------------------------------------------------
    // Geometry type coverage
    // ---------------------------------------------------------------------
    println!("--- Geometry type coverage ---\n");

    let inputs = [
        ("Point", "POINT(0 0)"),
        ("LineString", "LINESTRING(0 0, 5 0, 5 5)"),
        ("MultiPoint", "MULTIPOINT((0 0), (10 10))"),
        ("Polygon", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
        (
            "Polygon with hole",
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))",
        ),
        (
            "MultiPolygon",
            "MULTIPOLYGON(((0 0, 4 0, 4 4, 0 4, 0 0)), ((6 6, 9 6, 9 9, 6 9, 6 6)))",
        ),
    ];

    for (label, wkt) in inputs {
        let geom = Geometry::from_wkt(wkt).expect("valid WKT");
        match buffer(&geom, 1.0) {
            Ok(buffered) => println!("  {label:20} -> {}", buffered.geometry_type()),
            Err(e) => println!("  {label:20} -> error: {e}"),
        }
    }

    // ---------------------------------------------------------------------
    // Negative buffers (erosion)
    // ---------------------------------------------------------------------
    println!("\n--- Negative buffer (erosion) ---\n");

    let square = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").expect("valid WKT");
    for distance in [-1.0, -2.0, -4.0] {
        match buffer(&square, distance) {
            Ok(eroded) => println!(
                "  10x10 square eroded by {distance:>4} -> {}",
                eroded.to_wkt()
            ),
            Err(e) => println!("  10x10 square eroded by {distance:>4} -> error: {e}"),
        }
    }

    // ---------------------------------------------------------------------
    // Cap and join styles
    // ---------------------------------------------------------------------
    println!("\n--- Cap and join styles ---\n");

    let line = Geometry::from_wkt("LINESTRING(0 0, 5 0, 5 5)").expect("valid WKT");

    for cap in [CapStyle::Round, CapStyle::Flat, CapStyle::Square] {
        for join in [JoinStyle::Round, JoinStyle::Mitre, JoinStyle::Bevel] {
            let params = BufferParams {
                cap_style: cap,
                join_style: join,
                ..BufferParams::default()
            };
            match buffer_with_params(&line, 1.0, &params) {
                Ok(buffered) => {
                    let vertices = buffered.to_wkt().matches(',').count() + 1;
                    println!("  cap={cap:?}  join={join:?}  -> {vertices} vertices");
                }
                Err(e) => println!("  cap={cap:?}  join={join:?}  -> error: {e}"),
            }
        }
    }

    // ---------------------------------------------------------------------
    // Curve resolution
    // ---------------------------------------------------------------------
    println!("\n--- Curve resolution (quadrant_segments) ---\n");

    let point = Geometry::from_wkt("POINT(0 0)").expect("valid WKT");
    for segments in [2, 4, 8, 16] {
        let params = BufferParams {
            quadrant_segments: segments,
            ..BufferParams::default()
        };
        match buffer_with_params(&point, 1.0, &params) {
            Ok(buffered) => {
                let vertices = buffered.to_wkt().matches(',').count() + 1;
                println!("  quadrant_segments={segments:>2} -> {vertices} vertices");
            }
            Err(e) => println!("  quadrant_segments={segments:>2} -> error: {e}"),
        }
    }

    println!("\n=== Done ===");
}
