//! KML (Keyhole Markup Language) Support Example
//!
//! This example demonstrates the KML parsing and serialization capabilities
//! of the oxirs-geosparql crate. KML is the XML-based format used by Google Earth
//! and many other GIS applications.
//!
//! Run with: cargo run --example kml_support --features kml-support

#[cfg(feature = "kml-support")]
use oxirs_geosparql::geometry::kml_parser::{geometry_to_kml, parse_kml};
#[cfg(feature = "kml-support")]
use oxirs_geosparql::geometry::Geometry;

#[cfg(feature = "kml-support")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS GeoSPARQL: KML Support Demo ===\n");

    // ========================================
    // 1. Point Example
    // ========================================
    println!("1. Point Example");
    println!("   KML uses lon,lat,altitude format (note: different from typical lat,lon order)");

    let point_kml = r#"
        <Point>
            <coordinates>-122.0822035425683,37.42228990140251,0</coordinates>
        </Point>
    "#;

    let point_geometry = parse_kml(point_kml)?;
    println!("   Parsed: {}", point_geometry.to_wkt());
    println!("   CRS: {} (KML is always WGS84)", point_geometry.crs);

    // Convert back to KML
    let generated_kml = geometry_to_kml(&point_geometry)?;
    println!("   Generated KML:\n{}\n", generated_kml);

    // ========================================
    // 2. LineString Example (Google Earth Path)
    // ========================================
    println!("2. LineString Example (Path in Google Earth)");

    let linestring_kml = r#"
        <LineString>
            <coordinates>
                -122.08223,37.42254,0 -122.08219,37.42281,0 -122.08244,37.42292,0
            </coordinates>
        </LineString>
    "#;

    let linestring = parse_kml(linestring_kml)?;
    println!("   Parsed: {} vertices", linestring.to_wkt());

    // Convert back to KML
    let linestring_kml_out = geometry_to_kml(&linestring)?;
    println!("   Generated KML:\n{}\n", linestring_kml_out);

    // ========================================
    // 3. Polygon Example (Area in Google Earth)
    // ========================================
    println!("3. Polygon Example (Google Earth Area)");

    let polygon_kml = r#"
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        -122.084075,37.4220033612141,0
                        -122.085078,37.4220033612141,0
                        -122.085078,37.4229610810042,0
                        -122.084075,37.4229610810042,0
                        -122.084075,37.4220033612141,0
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    "#;

    let polygon = parse_kml(polygon_kml)?;
    println!("   Parsed polygon with {} vertices", polygon.to_wkt());

    // Convert back to KML
    let polygon_kml_out = geometry_to_kml(&polygon)?;
    println!("   Generated KML:\n{}\n", polygon_kml_out);

    // ========================================
    // 4. Polygon with Hole (Donut Shape)
    // ========================================
    println!("4. Polygon with Hole (Donut Shape)");

    let polygon_with_hole_kml = r#"
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>0,0,0 20,0,0 20,20,0 0,20,0 0,0,0</coordinates>
                </LinearRing>
            </outerBoundaryIs>
            <innerBoundaryIs>
                <LinearRing>
                    <coordinates>5,5,0 15,5,0 15,15,0 5,15,0 5,5,0</coordinates>
                </LinearRing>
            </innerBoundaryIs>
        </Polygon>
    "#;

    let polygon_with_hole = parse_kml(polygon_with_hole_kml)?;
    println!("   Parsed polygon with 1 hole");
    println!("   WKT: {}\n", polygon_with_hole.to_wkt());

    // ========================================
    // 5. MultiGeometry Example
    // ========================================
    println!("5. MultiGeometry Example (Collection of Points)");

    let multi_geometry_kml = r#"
        <MultiGeometry>
            <Point>
                <coordinates>-122.0822,37.4223,0</coordinates>
            </Point>
            <Point>
                <coordinates>-122.0844,37.4228,0</coordinates>
            </Point>
            <Point>
                <coordinates>-122.0856,37.4235,0</coordinates>
            </Point>
        </MultiGeometry>
    "#;

    let multi_geometry = parse_kml(multi_geometry_kml)?;
    println!("   Parsed: {}", multi_geometry.geometry_type());
    println!("   WKT: {}\n", multi_geometry.to_wkt());

    // ========================================
    // 6. Creating KML from Rust Geometry
    // ========================================
    println!("6. Creating KML from Rust Geometry Objects");

    use geo_types::{Geometry as GeoGeometry, LineString, Point, Polygon};

    // Create a point
    let rust_point = Point::new(-122.0822035425683, 37.42228990140251);
    let point_geom = Geometry::new(GeoGeometry::Point(rust_point));
    let point_kml = geometry_to_kml(&point_geom)?;
    println!("   Point to KML:\n{}", point_kml);

    // Create a polygon
    let exterior = LineString::from(vec![
        (-122.084075, 37.4220033612141),
        (-122.085078, 37.4220033612141),
        (-122.085078, 37.4229610810042),
        (-122.084075, 37.4229610810042),
        (-122.084075, 37.4220033612141),
    ]);
    let rust_polygon = Polygon::new(exterior, vec![]);
    let polygon_geom = Geometry::new(GeoGeometry::Polygon(rust_polygon));
    let polygon_kml = geometry_to_kml(&polygon_geom)?;
    println!("   Polygon to KML:\n{}\n", polygon_kml);

    // ========================================
    // 7. Round-Trip Conversion Test
    // ========================================
    println!("7. Round-Trip Conversion Test");

    let original_kml = r#"
        <Point>
            <coordinates>-122.0822035425683,37.42228990140251,0</coordinates>
        </Point>
    "#;

    let parsed = parse_kml(original_kml)?;
    let regenerated_kml = geometry_to_kml(&parsed)?;
    let reparsed = parse_kml(&regenerated_kml)?;

    println!("   Original: {}", parsed.to_wkt());
    println!("   After round-trip: {}", reparsed.to_wkt());
    println!("   Match: {}\n", parsed.to_wkt() == reparsed.to_wkt());

    // ========================================
    // 8. Complex Example: Multiple Features
    // ========================================
    println!("8. Complex Example: Simulating Google Earth Placemark");
    println!(
        "   (In real KML, this would be wrapped in <Placemark> tags with name, description, etc.)"
    );

    let placemark_kml = r#"
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        -122.084893,37.422571,17
                        -122.084902,37.422119,17
                        -122.084520,37.422109,17
                        -122.084520,37.422557,17
                        -122.084893,37.422571,17
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    "#;

    let placemark = parse_kml(placemark_kml)?;
    println!(
        "   Parsed placemark geometry: {}",
        placemark.geometry_type()
    );
    println!("   WKT: {}\n", placemark.to_wkt());

    // ========================================
    // 9. CRS Information
    // ========================================
    println!("9. Coordinate Reference System (CRS)");
    println!("   KML always uses WGS84 (EPSG:4326) coordinate system");
    println!("   Coordinates are in lon,lat,altitude order");
    println!("   CRS URI: {}\n", point_geometry.crs.uri);

    // ========================================
    // 10. Integration with Other Formats
    // ========================================
    println!("10. Converting Between Formats");

    let kml_point = r#"
        <Point>
            <coordinates>-122.08,37.42,0</coordinates>
        </Point>
    "#;

    let geom = parse_kml(kml_point)?;
    println!("    KML -> WKT: {}", geom.to_wkt());

    #[cfg(feature = "gml-support")]
    {
        use oxirs_geosparql::geometry::gml_parser::geometry_to_gml;
        let gml = geometry_to_gml(&geom)?;
        println!("    KML -> GML:\n{}", gml);
    }

    #[cfg(feature = "geojson-support")]
    {
        let geojson = geom.to_geojson()?;
        println!("    KML -> GeoJSON: {}", geojson);
    }

    // ========================================
    // Summary
    // ========================================
    println!("\n=== Summary ===");
    println!("KML (Keyhole Markup Language) is the XML format used by Google Earth.");
    println!();
    println!("Key Features:");
    println!("  • XML-based format (uses quick-xml for parsing)");
    println!("  • Always uses WGS84 coordinate system (EPSG:4326)");
    println!("  • Coordinates in lon,lat,altitude order");
    println!("  • Supports: Point, LineString, Polygon, MultiGeometry");
    println!("  • Handles polygon holes (innerBoundaryIs)");
    println!("  • Compatible with Google Earth and many GIS tools");
    println!();
    println!("Usage in OxiRS:");
    println!("  • Enable with feature flag: --features kml-support");
    println!("  • Parse: Geometry::from_kml(kml_string)");
    println!("  • Serialize: geometry.to_kml()");
    println!("  • Round-trip conversion supported");

    Ok(())
}

#[cfg(not(feature = "kml-support"))]
fn main() {
    eprintln!("This example requires the 'kml-support' feature.");
    eprintln!("Run with: cargo run --example kml_support --features kml-support");
    std::process::exit(1);
}
