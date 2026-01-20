//! Coordinate Reference System (CRS) Transformation Example
//!
//! This example demonstrates how to transform geometries between different
//! Coordinate Reference Systems using the PROJ library.
//!
//! Run with: cargo run --example crs_transformation --features proj-support

use oxirs_geosparql::error::Result;

#[cfg(feature = "proj-support")]
use oxirs_geosparql::geometry::{Crs, Geometry};

#[cfg(feature = "proj-support")]
use oxirs_geosparql::functions::coordinate_transformation::{transform, transform_batch};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL CRS Transformation Example ===\n");

    #[cfg(feature = "proj-support")]
    {
        // Example 1: Transform a point from WGS84 to Web Mercator
        println!("1. POINT TRANSFORMATION (WGS84 → Web Mercator):");
        println!("   Common use case: Web mapping applications (Google Maps, OpenStreetMap)\n");

        let tokyo_wgs84 = Geometry::from_wkt(
            "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(139.6917 35.6895)",
        )?;
        println!("   Tokyo (WGS84/EPSG:4326): {}", tokyo_wgs84.to_wkt());

        let tokyo_mercator = transform(&tokyo_wgs84, &Crs::epsg(3857))?;
        println!(
            "   Tokyo (Web Mercator/EPSG:3857): {}",
            tokyo_mercator.to_wkt()
        );
        println!("   ✅ Transformed for web mapping\n");

        // Example 2: Transform between different geographic CRS
        println!("2. GEOGRAPHIC CRS TRANSFORMATION (WGS84 → NAD83):");
        println!("   Use case: Converting between global and regional reference systems\n");

        let new_york = Geometry::from_wkt(
            "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(-74.006 40.7128)",
        )?;
        println!("   New York (WGS84/EPSG:4326): {}", new_york.to_wkt());

        // NAD83 is EPSG:4269
        let new_york_nad83 = transform(&new_york, &Crs::epsg(4269))?;
        println!("   New York (NAD83/EPSG:4269): {}", new_york_nad83.to_wkt());
        println!("   ✅ Regional reference system\n");

        // Example 3: Transform a LineString
        println!("3. LINESTRING TRANSFORMATION:");
        println!("   Use case: Converting road or boundary data\n");

        let route = Geometry::from_wkt(
            "<http://www.opengis.net/def/crs/EPSG/0/4326> LINESTRING(139.7 35.7, 140.0 36.0, 140.5 36.5)",
        )?;
        println!("   Route (WGS84): {}", route.to_wkt());

        let route_mercator = transform(&route, &Crs::epsg(3857))?;
        println!("   Route (Web Mercator): {}", route_mercator.to_wkt());
        println!("   ✅ All coordinates transformed\n");

        // Example 4: Transform a Polygon
        println!("4. POLYGON TRANSFORMATION:");
        println!("   Use case: Administrative boundaries, land parcels\n");

        let area = Geometry::from_wkt(
            "<http://www.opengis.net/def/crs/EPSG/0/4326> POLYGON((139 35, 140 35, 140 36, 139 36, 139 35))",
        )?;
        println!("   Area (WGS84): {}", area.to_wkt());

        let area_mercator = transform(&area, &Crs::epsg(3857))?;
        println!("   Area (Web Mercator): {}", area_mercator.to_wkt());
        println!("   ✅ Polygon boundary transformed\n");

        // Example 5: Batch transformation
        println!("5. BATCH TRANSFORMATION:");
        println!("   Use case: Transforming multiple geometries efficiently\n");

        let cities_wgs84 = vec![
            Geometry::from_wkt(
                "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(139.6917 35.6895)",
            )?, // Tokyo
            Geometry::from_wkt(
                "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(126.9780 37.5665)",
            )?, // Seoul
            Geometry::from_wkt(
                "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(114.1095 22.3964)",
            )?, // Hong Kong
        ];

        println!(
            "   Transforming {} cities from WGS84 to Web Mercator:",
            cities_wgs84.len()
        );
        let cities_mercator = transform_batch(&cities_wgs84, &Crs::epsg(3857))?;

        for (i, city) in cities_mercator.iter().enumerate() {
            println!("   City {}: {}", i + 1, city.to_wkt());
        }
        println!("   ✅ Batch transformation complete\n");

        // Example 6: Round-trip transformation
        println!("6. ROUND-TRIP TRANSFORMATION:");
        println!("   Verify transformation accuracy by converting back\n");

        let original =
            Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(100.0 30.0)")?;
        println!("   Original (WGS84): {}", original.to_wkt());

        let mercator = transform(&original, &Crs::epsg(3857))?;
        println!("   Transformed (Web Mercator): {}", mercator.to_wkt());

        let back_to_wgs84 = transform(&mercator, &Crs::epsg(4326))?;
        println!("   Back to WGS84: {}", back_to_wgs84.to_wkt());
        println!("   ✅ Round-trip transformation\n");

        // Example 7: Real-world scenario - UTM zones
        println!("7. UTM ZONE TRANSFORMATION:");
        println!("   Use case: High-precision local coordinate systems\n");

        // Tokyo is in UTM Zone 54N (EPSG:32654)
        let tokyo = Geometry::from_wkt(
            "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(139.6917 35.6895)",
        )?;
        println!("   Tokyo (WGS84): {}", tokyo.to_wkt());

        let tokyo_utm = transform(&tokyo, &Crs::epsg(32654))?;
        println!("   Tokyo (UTM 54N): {}", tokyo_utm.to_wkt());
        println!("   ✅ UTM coordinates for local surveying\n");

        // Example 8: Error handling - non-EPSG CRS
        println!("8. ERROR HANDLING:");
        println!("   CRS transformation requires EPSG codes\n");

        let geom_with_custom_crs = Geometry::with_crs(
            geo_types::Geometry::Point(geo_types::Point::new(139.7, 35.7)),
            Crs::new("http://www.opengis.net/def/crs/OGC/1.3/CRS84"),
        );

        match transform(&geom_with_custom_crs, &Crs::epsg(3857)) {
            Ok(_) => println!("   Unexpected success"),
            Err(e) => println!("   ✅ Correctly caught error: {}", e),
        }
        println!();

        // Summary
        println!("\n=== COMMON EPSG CODES ===\n");
        println!("Geographic Coordinate Systems:");
        println!("  • EPSG:4326  - WGS84 (GPS standard, global)");
        println!("  • EPSG:4269  - NAD83 (North America)");
        println!("  • EPSG:4277  - OSGB36 (Great Britain)");
        println!("  • EPSG:4258  - ETRS89 (Europe)\n");

        println!("Projected Coordinate Systems:");
        println!("  • EPSG:3857  - Web Mercator (Google Maps, OSM)");
        println!("  • EPSG:32601-32660 - UTM Northern Hemisphere (zones 1-60)");
        println!("  • EPSG:32701-32760 - UTM Southern Hemisphere (zones 1-60)");
        println!("  • EPSG:2154  - Lambert 93 (France)");
        println!("  • EPSG:27700 - British National Grid\n");

        println!("=== USE CASES ===\n");
        println!("1. **Web Mapping**: WGS84 → Web Mercator (EPSG:3857)");
        println!("   - Required for Google Maps, OpenStreetMap, Mapbox");
        println!("   - Fast tiled rendering\n");

        println!("2. **GPS Data Processing**: Any CRS → WGS84 (EPSG:4326)");
        println!("   - Standard GPS coordinate system");
        println!("   - Global interoperability\n");

        println!("3. **Local Surveying**: WGS84 → UTM");
        println!("   - High precision for local areas");
        println!("   - Metric units (meters)\n");

        println!("4. **Cross-border Analysis**: Multiple CRS → Common CRS");
        println!("   - Combining datasets from different countries");
        println!("   - Ensuring spatial accuracy\n");

        println!("=== PERFORMANCE TIPS ===\n");
        println!("1. Use `transform_batch()` for multiple geometries");
        println!("2. Transform once, cache results");
        println!("3. Choose appropriate CRS for your use case");
        println!("4. Verify transformation accuracy with round-trip tests\n");

        println!("=== Example completed successfully! ===");
    }

    #[cfg(not(feature = "proj-support"))]
    {
        println!("❌ This example requires the 'proj-support' feature to be enabled.\n");
        println!("Please run with:");
        println!("   cargo run --example crs_transformation --features proj-support\n");
        println!("Note: This requires PROJ library to be installed:");
        println!("  • macOS: brew install proj");
        println!("  • Ubuntu: sudo apt-get install libproj-dev");
        println!("  • Fedora: sudo dnf install proj-devel");
    }

    Ok(())
}
