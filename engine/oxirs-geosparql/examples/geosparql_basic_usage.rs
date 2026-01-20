//! Basic usage example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Parsing WKT geometries
//! - Creating geometries programmatically
//! - Converting geometries to WKT
//! - Basic geometry properties
//!
//! Run with: cargo run --example basic_usage

use oxirs_geosparql::error::Result;
use oxirs_geosparql::geometry::{Crs, Geometry};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Basic Usage Example ===\n");

    // 1. Parse WKT geometries
    println!("1. Parsing WKT geometries:");
    let point = Geometry::from_wkt("POINT(139.6917 35.6895)")?;
    println!("  Point: {}", point.to_wkt());
    println!("  Type: {}", point.geometry_type());
    println!("  Dimension: {}", point.dimension());
    println!("  Spatial dimension: {}", point.spatial_dimension());

    let linestring = Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 2, 3 3)")?;
    println!("\n  LineString: {}", linestring.to_wkt());
    println!("  Type: {}", linestring.geometry_type());

    let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    println!("\n  Polygon: {}", polygon.to_wkt());
    println!("  Type: {}", polygon.geometry_type());
    println!("  Empty: {}", polygon.is_empty());

    // 2. Parse WKT with CRS
    println!("\n2. Parsing WKT with CRS:");
    let point_with_crs =
        Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(0 51.5)")?;
    println!("  Point: {}", point_with_crs.to_wkt());
    println!("  CRS: {}", point_with_crs.crs);
    println!("  Is default CRS: {}", point_with_crs.crs.is_default());

    // 3. Create geometry with specific CRS
    println!("\n3. Creating geometry with specific CRS:");
    let _wgs84_point = Geometry::from_wkt("POINT(-74.0060 40.7128)")?;
    let epsg_4326_crs = Crs::epsg(4326);
    println!("  EPSG:4326 CRS URI: {}", epsg_4326_crs);

    // 4. Multi-geometries
    println!("\n4. Multi-geometries:");
    let multipoint = Geometry::from_wkt("MULTIPOINT((0 0), (1 1), (2 2))")?;
    println!("  MultiPoint: {}", multipoint.to_wkt());

    let multilinestring = Geometry::from_wkt("MULTILINESTRING((0 0, 1 1), (2 2, 3 3))")?;
    println!("  MultiLineString: {}", multilinestring.to_wkt());

    let multipolygon = Geometry::from_wkt(
        "MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)), ((10 10, 15 10, 15 15, 10 15, 10 10)))",
    )?;
    println!("  MultiPolygon: {}", multipolygon.to_wkt());

    // 5. Empty geometries
    println!("\n5. Empty geometries:");
    let empty_point = Geometry::from_wkt("POINT EMPTY")?;
    println!("  Empty point: {}", empty_point.to_wkt());
    println!("  Is empty: {}", empty_point.is_empty());

    // 6. Geometry properties
    println!("\n6. Geometry properties:");
    println!("  Point is 3D: {}", point.is_3d());
    println!("  Point is measured: {}", point.is_measured());
    println!("  Point is simple: {}", point.is_simple());

    // 7. Round-trip conversion
    println!("\n7. Round-trip WKT conversion:");
    let original_wkt = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))";
    let geom = Geometry::from_wkt(original_wkt)?;
    let new_wkt = geom.to_wkt();
    let geom2 = Geometry::from_wkt(&new_wkt)?;
    println!("  Original: {}", original_wkt);
    println!("  After round-trip: {}", geom2.to_wkt());
    println!(
        "  Types match: {}",
        geom.geometry_type() == geom2.geometry_type()
    );

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
