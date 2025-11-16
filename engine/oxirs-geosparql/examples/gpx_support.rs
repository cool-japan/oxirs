//! GPX (GPS Exchange Format) Support Example
//!
//! This example demonstrates the GPX parsing and serialization capabilities
//! of the oxirs-geosparql crate. GPX is the XML-based format used by GPS devices
//! and many fitness/navigation applications.
//!
//! Run with: cargo run --example gpx_support --features gpx-support

#[cfg(feature = "gpx-support")]
use oxirs_geosparql::geometry::gpx_parser::{geometry_to_gpx, parse_gpx};
#[cfg(feature = "gpx-support")]
use oxirs_geosparql::geometry::Geometry;

#[cfg(feature = "gpx-support")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS GeoSPARQL: GPX Support Demo ===\n");

    // ========================================
    // 1. Waypoint Example (Single Point)
    // ========================================
    println!("1. Waypoint Example (GPS Point of Interest)");
    println!("   GPX uses lat/lon attributes (note: typical lat,lon order)");

    let waypoint_gpx = r#"
        <wpt lat="37.422" lon="-122.084">
            <name>Googleplex</name>
            <desc>Google Headquarters</desc>
        </wpt>
    "#;

    let waypoint = parse_gpx(waypoint_gpx)?;
    println!("   Parsed: {}", waypoint.to_wkt());
    println!("   CRS: {} (GPX is always WGS84)", waypoint.crs);

    // Convert back to GPX
    let generated_gpx = geometry_to_gpx(&waypoint, Some("Googleplex"))?;
    println!("   Generated GPX:\n{}\n", generated_gpx);

    // ========================================
    // 2. Track Example (GPS Recording)
    // ========================================
    println!("2. Track Example (GPS Recording/Trail)");
    println!("   Tracks represent recorded paths from GPS devices");

    let track_gpx = r#"
        <trk>
            <name>Morning Run</name>
            <trkseg>
                <trkpt lat="37.422" lon="-122.084">
                    <ele>10</ele>
                    <time>2025-10-31T08:00:00Z</time>
                </trkpt>
                <trkpt lat="37.423" lon="-122.085">
                    <ele>12</ele>
                    <time>2025-10-31T08:05:00Z</time>
                </trkpt>
                <trkpt lat="37.424" lon="-122.086">
                    <ele>15</ele>
                    <time>2025-10-31T08:10:00Z</time>
                </trkpt>
            </trkseg>
        </trk>
    "#;

    let track = parse_gpx(track_gpx)?;
    println!("   Parsed: {} with 3 track points", track.geometry_type());
    println!("   WKT: {}\n", track.to_wkt());

    // ========================================
    // 3. Route Example (Planned Path)
    // ========================================
    println!("3. Route Example (Planned Navigation Route)");
    println!("   Routes represent planned paths (vs tracks which are recorded)");

    let route_gpx = r#"
        <rte>
            <name>Bike Route to Park</name>
            <rtept lat="37.422" lon="-122.084">
                <name>Start</name>
            </rtept>
            <rtept lat="37.425" lon="-122.087">
                <name>Turn Point</name>
            </rtept>
            <rtept lat="37.428" lon="-122.090">
                <name>Park</name>
            </rtept>
        </rte>
    "#;

    let route = parse_gpx(route_gpx)?;
    println!("   Parsed: {} with 3 route points", route.geometry_type());
    println!("   WKT: {}\n", route.to_wkt());

    // ========================================
    // 4. Creating GPX from Rust Geometry
    // ========================================
    println!("4. Creating GPX from Rust Geometry Objects");

    use geo_types::{Geometry as GeoGeometry, LineString, Point};

    // Create a waypoint
    let rust_point = Point::new(-122.084, 37.422);
    let point_geom = Geometry::new(GeoGeometry::Point(rust_point));
    let point_gpx = geometry_to_gpx(&point_geom, Some("Coffee Shop"))?;
    println!("   Waypoint GPX:\n{}", point_gpx);

    // Create a track
    let track_line = LineString::from(vec![
        (-122.084, 37.422),
        (-122.085, 37.423),
        (-122.086, 37.424),
        (-122.087, 37.425),
    ]);
    let track_geom = Geometry::new(GeoGeometry::LineString(track_line));
    let track_gpx = geometry_to_gpx(&track_geom, Some("Evening Walk"))?;
    println!("   Track GPX:\n{}\n", track_gpx);

    // ========================================
    // 5. Round-Trip Conversion Test
    // ========================================
    println!("5. Round-Trip Conversion Test");

    let original_gpx = r#"<wpt lat="37.422" lon="-122.084"><name>Test</name></wpt>"#;
    let parsed = parse_gpx(original_gpx)?;
    let regenerated_gpx = geometry_to_gpx(&parsed, Some("Test"))?;
    let reparsed = parse_gpx(&regenerated_gpx)?;

    println!("   Original: {}", parsed.to_wkt());
    println!("   After round-trip: {}", reparsed.to_wkt());
    println!("   Match: {}\n", parsed.to_wkt() == reparsed.to_wkt());

    // ========================================
    // 6. CRS Information
    // ========================================
    println!("6. Coordinate Reference System (CRS)");
    println!("   GPX always uses WGS84 (EPSG:4326) coordinate system");
    println!("   Coordinates are in lat/lon order (standard GPS format)");
    println!("   CRS URI: {}\n", waypoint.crs.uri);

    // ========================================
    // 7. Integration with Other Formats
    // ========================================
    println!("7. Converting Between Formats");

    let gpx_wpt = r#"<wpt lat="37.422" lon="-122.084"/>"#;
    let geom = parse_gpx(gpx_wpt)?;
    println!("   GPX -> WKT: {}", geom.to_wkt());

    #[cfg(feature = "kml-support")]
    {
        let kml = geom.to_kml()?;
        println!("   GPX -> KML:\n{}", kml);
    }

    #[cfg(feature = "geojson-support")]
    {
        let geojson = geom.to_geojson()?;
        println!("   GPX -> GeoJSON: {}", geojson);
    }

    // ========================================
    // 8. Real-World Use Cases
    // ========================================
    println!("\n8. Real-World Use Cases");

    // Use case 1: Fitness tracking
    println!("\n   Use Case 1: Fitness Activity Tracking");
    let fitness_track = LineString::from(vec![
        (-122.084, 37.422),
        (-122.085, 37.423),
        (-122.086, 37.424),
        (-122.087, 37.425),
        (-122.088, 37.426),
    ]);
    let fitness_geom = Geometry::new(GeoGeometry::LineString(fitness_track));
    let _fitness_gpx = geometry_to_gpx(&fitness_geom, Some("5K Run"))?;
    println!("   Created GPX track for fitness app");
    println!("   Track has {} points", 5);

    // Use case 2: Geocaching
    println!("\n   Use Case 2: Geocaching Waypoints");
    let cache_point = Point::new(-122.084, 37.422);
    let cache_geom = Geometry::new(GeoGeometry::Point(cache_point));
    let _cache_gpx = geometry_to_gpx(&cache_geom, Some("Hidden Cache #42"))?;
    println!("   Created waypoint for geocache location");

    // Use case 3: Hiking trail
    println!("\n   Use Case 3: Hiking Trail Map");
    let trail = LineString::from(vec![
        (-122.084, 37.422), // Trailhead
        (-122.086, 37.424), // Scenic overlook
        (-122.088, 37.426), // Water source
        (-122.090, 37.428), // Summit
    ]);
    let trail_geom = Geometry::new(GeoGeometry::LineString(trail));
    let _trail_gpx = geometry_to_gpx(&trail_geom, Some("Mount Davidson Trail"))?;
    println!("   Created trail route with {} waypoints", 4);

    // ========================================
    // 9. GPX vs Other Formats
    // ========================================
    println!("\n9. GPX vs Other GPS Formats");
    println!("   GPX:     XML-based, widely supported, human-readable");
    println!("   KML:     XML-based, Google Earth format, richer styling");
    println!("   GeoJSON: JSON-based, web-friendly, modern");
    println!("   WKT:     Text-based, database-friendly, simple");

    // ========================================
    // Summary
    // ========================================
    println!("\n=== Summary ===");
    println!("GPX (GPS Exchange Format) is the standard XML format for GPS data.");
    println!();
    println!("Key Features:");
    println!("  • XML-based format (uses quick-xml for parsing)");
    println!("  • Always uses WGS84 coordinate system (EPSG:4326)");
    println!("  • Coordinates in lat/lon attribute format");
    println!("  • Supports: Waypoints (Points), Tracks (LineStrings), Routes (LineStrings)");
    println!("  • Used by: Garmin, Strava, AllTrails, most GPS devices");
    println!("  • Can include metadata: names, descriptions, elevation, time");
    println!();
    println!("Usage in OxiRS:");
    println!("  • Enable with feature flag: --features gpx-support");
    println!("  • Parse: Geometry::from_gpx(gpx_string)");
    println!("  • Serialize: geometry.to_gpx(optional_name)");
    println!("  • Round-trip conversion supported");
    println!();
    println!("Typical Applications:");
    println!("  • Fitness tracking apps (Strava, RunKeeper)");
    println!("  • Hiking/trail navigation");
    println!("  • Geocaching");
    println!("  • GPS device data exchange");
    println!("  • Route planning and sharing");

    Ok(())
}

#[cfg(not(feature = "gpx-support"))]
fn main() {
    eprintln!("This example requires the 'gpx-support' feature.");
    eprintln!("Run with: cargo run --example gpx_support --features gpx-support");
    std::process::exit(1);
}
