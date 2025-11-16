//! FlatGeobuf support example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Writing geometries to FlatGeobuf format
//! - Reading geometries from FlatGeobuf files
//! - Round-trip conversion (Geometry → FlatGeobuf → Geometry)
//! - Working with various geometry types in FlatGeobuf
//! - Performance benefits of binary format
//!
//! Run with: cargo run --example flatgeobuf_support --features flatgeobuf-support

use oxirs_geosparql::error::Result;

#[cfg(feature = "flatgeobuf-support")]
use oxirs_geosparql::geometry::Geometry;

#[cfg(feature = "flatgeobuf-support")]
use geo_types::{Coord, Geometry as GeoGeometry, LineString, MultiPoint, Point, Polygon};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL FlatGeobuf Support Example ===\n");

    #[cfg(feature = "flatgeobuf-support")]
    {
        use std::env;
        use std::fs::File;
        use std::io::{BufReader, BufWriter};

        let temp_dir = env::temp_dir();

        // 1. Writing and reading Points
        println!("1. POINT GEOMETRY - Binary Format:");
        let point = Geometry::new(GeoGeometry::Point(Point::new(125.6, 10.1)));
        println!("   WKT: {}", point.to_wkt());

        let point_path = temp_dir.join("point.fgb");
        let file = File::create(&point_path)?;
        let writer = BufWriter::new(file);
        Geometry::to_flatgeobuf(std::slice::from_ref(&point), writer)?;
        println!("   ✓ Written to: {}", point_path.display());

        let file = File::open(&point_path)?;
        let reader = BufReader::new(file);
        let geometries = Geometry::from_flatgeobuf(reader)?;
        println!("   ✓ Read back: {} geometries", geometries.len());
        println!("   WKT: {}", geometries[0].to_wkt());

        // 2. LineString (Route)
        println!("\n2. LINESTRING GEOMETRY - Route Data:");
        let route = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 102.0, y: 0.0 },
            Coord { x: 103.0, y: 1.0 },
            Coord { x: 104.0, y: 0.0 },
            Coord { x: 105.0, y: 1.0 },
        ])));
        println!("   WKT: {}", route.to_wkt());

        let route_path = temp_dir.join("route.fgb");
        Geometry::to_flatgeobuf(std::slice::from_ref(&route), File::create(&route_path)?)?;
        println!("   ✓ Written to: {}", route_path.display());

        let geometries = Geometry::from_flatgeobuf(BufReader::new(File::open(&route_path)?))?;
        println!("   ✓ Read back: {} geometries", geometries.len());

        // 3. Polygon (Area)
        println!("\n3. POLYGON GEOMETRY - Area Boundary:");
        let polygon = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 100.0, y: 0.0 },
                Coord { x: 101.0, y: 0.0 },
                Coord { x: 101.0, y: 1.0 },
                Coord { x: 100.0, y: 1.0 },
                Coord { x: 100.0, y: 0.0 },
            ]),
            vec![],
        )));
        println!("   WKT: {}", polygon.to_wkt());

        let poly_path = temp_dir.join("polygon.fgb");
        Geometry::to_flatgeobuf(std::slice::from_ref(&polygon), File::create(&poly_path)?)?;
        println!("   ✓ Written to: {}", poly_path.display());

        let geometries = Geometry::from_flatgeobuf(BufReader::new(File::open(&poly_path)?))?;
        println!("   ✓ Read back: {} geometries", geometries.len());
        println!("   WKT: {}", geometries[0].to_wkt());

        // 4. MultiPoint (Collection of locations)
        println!("\n4. MULTIPOINT GEOMETRY - Multiple Locations:");
        let multipoint = Geometry::new(GeoGeometry::MultiPoint(MultiPoint::new(vec![
            Point::new(10.0, 20.0),
            Point::new(15.0, 25.0),
            Point::new(20.0, 30.0),
        ])));
        println!("   WKT: {}", multipoint.to_wkt());

        let mp_path = temp_dir.join("multipoint.fgb");
        Geometry::to_flatgeobuf(std::slice::from_ref(&multipoint), File::create(&mp_path)?)?;
        println!("   ✓ Written to: {}", mp_path.display());

        let geometries = Geometry::from_flatgeobuf(BufReader::new(File::open(&mp_path)?))?;
        println!("   ✓ Read back: {} geometries", geometries.len());

        // 5. Multiple geometries in one file
        println!("\n5. MULTIPLE GEOMETRIES - Batch Processing:");
        let points = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
            Geometry::new(GeoGeometry::Point(Point::new(5.0, 6.0))),
        ];
        println!("   Writing {} points...", points.len());

        let batch_path = temp_dir.join("batch.fgb");
        Geometry::to_flatgeobuf(&points, File::create(&batch_path)?)?;
        println!("   ✓ Written to: {}", batch_path.display());

        let geometries = Geometry::from_flatgeobuf(BufReader::new(File::open(&batch_path)?))?;
        println!("   ✓ Read back: {} geometries", geometries.len());
        for (i, geom) in geometries.iter().enumerate() {
            println!("     Point {}: {}", i + 1, geom.to_wkt());
        }

        // 6. Format comparison
        println!("\n6. FORMAT COMPARISON:");
        let test_geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));

        // FlatGeobuf binary size
        let fgb_path = temp_dir.join("test.fgb");
        Geometry::to_flatgeobuf(std::slice::from_ref(&test_geom), File::create(&fgb_path)?)?;
        let fgb_size = std::fs::metadata(&fgb_path)?.len();
        println!("   FlatGeobuf (.fgb) size: {} bytes", fgb_size);

        // WKT text size
        let wkt = test_geom.to_wkt();
        println!("   WKT text size: {} bytes", wkt.len());
        println!("   WKT: {}", wkt);

        #[cfg(feature = "geojson-support")]
        {
            let geojson = test_geom.to_geojson()?;
            println!("   GeoJSON size: {} bytes", geojson.len());
        }

        println!("\n7. ADVANTAGES OF FLATGEOBUF:");
        println!("   ✓ Compact binary format");
        println!("   ✓ Fast serialization/deserialization");
        println!("   ✓ Built-in spatial indexing support");
        println!("   ✓ HTTP range request friendly (cloud-native)");
        println!("   ✓ Streaming read capability");
        println!("   ✓ Smaller file sizes than GeoJSON");
        println!("   ✓ Maintains precision better than text formats");

        println!("\n8. USE CASES:");
        println!("   • Large-scale GIS data storage");
        println!("   • Cloud-native geospatial applications");
        println!("   • Streaming spatial data processing");
        println!("   • Web mapping tile servers");
        println!("   • Distributed spatial databases");
        println!("   • Mobile GIS applications (smaller downloads)");

        // Cleanup
        let _ = std::fs::remove_file(&point_path);
        let _ = std::fs::remove_file(&route_path);
        let _ = std::fs::remove_file(&poly_path);
        let _ = std::fs::remove_file(&mp_path);
        let _ = std::fs::remove_file(&batch_path);
        let _ = std::fs::remove_file(&fgb_path);

        println!("\n✓ All tests passed!");
    }

    #[cfg(not(feature = "flatgeobuf-support"))]
    {
        println!("FlatGeobuf support is not enabled!");
        println!("Run with: cargo run --example flatgeobuf_support --features flatgeobuf-support");
    }

    Ok(())
}
