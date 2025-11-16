//! GeoJSON support example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Parsing GeoJSON geometries and feature collections
//! - Converting between GeoJSON, WKT, and internal representations
//! - Handling properties and feature metadata
//! - Round-trip conversions (GeoJSON → Geometry → GeoJSON)
//! - Working with various geometry types in GeoJSON
//!
//! Run with: cargo run --example geojson_support --features geojson-support

use oxirs_geosparql::error::Result;

#[cfg(feature = "geojson-support")]
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL GeoJSON Support Example ===\n");

    #[cfg(feature = "geojson-support")]
    {
        // 1. Basic Point parsing
        println!("1. BASIC POINT PARSING:");
        let geojson_point = r#"{
            "type": "Point",
            "coordinates": [125.6, 10.1]
        }"#;

        let point = Geometry::from_geojson(geojson_point)?;
        println!("   GeoJSON Input: {}", geojson_point.replace('\n', ""));
        println!("   Parsed WKT: {}", point.to_wkt());
        println!("   Back to GeoJSON: {}", point.to_geojson()?);

        // 2. LineString
        println!("\n2. LINESTRING (PATH/ROUTE):");
        let geojson_linestring = r#"{
            "type": "LineString",
            "coordinates": [
                [102.0, 0.0],
                [103.0, 1.0],
                [104.0, 0.0],
                [105.0, 1.0]
            ]
        }"#;

        let route = Geometry::from_geojson(geojson_linestring)?;
        println!("   GeoJSON Input: {}", geojson_linestring.replace('\n', ""));
        println!("   Parsed WKT: {}", route.to_wkt());
        println!("   Geometry Type: {}", route.geometry_type());

        // 3. Polygon (Area)
        println!("\n3. POLYGON (AREA/BOUNDARY):");
        let geojson_polygon = r#"{
            "type": "Polygon",
            "coordinates": [
                [
                    [100.0, 0.0],
                    [101.0, 0.0],
                    [101.0, 1.0],
                    [100.0, 1.0],
                    [100.0, 0.0]
                ]
            ]
        }"#;

        let area = Geometry::from_geojson(geojson_polygon)?;
        println!("   GeoJSON Input: {}", geojson_polygon.replace('\n', ""));
        println!("   Parsed WKT: {}", area.to_wkt());
        println!("   Spatial Dimension: {}", area.spatial_dimension());

        // 4. Polygon with hole
        println!("\n4. POLYGON WITH HOLE (DONUT SHAPE):");
        let geojson_polygon_hole = r#"{
            "type": "Polygon",
            "coordinates": [
                [
                    [100.0, 0.0],
                    [101.0, 0.0],
                    [101.0, 1.0],
                    [100.0, 1.0],
                    [100.0, 0.0]
                ],
                [
                    [100.2, 0.2],
                    [100.8, 0.2],
                    [100.8, 0.8],
                    [100.2, 0.8],
                    [100.2, 0.2]
                ]
            ]
        }"#;

        let donut = Geometry::from_geojson(geojson_polygon_hole)?;
        println!(
            "   GeoJSON Input: {}",
            geojson_polygon_hole.replace('\n', "")
        );
        println!("   Parsed WKT: {}", donut.to_wkt());
        println!("   Description: Outer boundary with inner exclusion");

        // 5. MultiPoint
        println!("\n5. MULTIPOINT (COLLECTION OF LOCATIONS):");
        let geojson_multipoint = r#"{
            "type": "MultiPoint",
            "coordinates": [
                [100.0, 0.0],
                [101.0, 1.0],
                [102.0, 2.0]
            ]
        }"#;

        let locations = Geometry::from_geojson(geojson_multipoint)?;
        println!("   GeoJSON Input: {}", geojson_multipoint.replace('\n', ""));
        println!("   Parsed WKT: {}", locations.to_wkt());
        println!("   Use case: Store locations, sensor positions");

        // 6. Feature with properties
        println!("\n6. FEATURE WITH PROPERTIES:");
        let geojson_feature = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [-122.4194, 37.7749]
            },
            "properties": {
                "name": "San Francisco",
                "population": 884363,
                "state": "California"
            }
        }"#;

        let feature = Geometry::from_geojson(geojson_feature)?;
        println!("   Feature Location: San Francisco");
        println!("   Parsed WKT: {}", feature.to_wkt());
        println!("   Geometry Type: {}", feature.geometry_type());

        // 7. Feature Collection
        println!("\n7. FEATURE COLLECTION (MULTIPLE LOCATIONS):");
        let geojson_collection = r#"{
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-122.4194, 37.7749]
                    },
                    "properties": {"name": "San Francisco"}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-118.2437, 34.0522]
                    },
                    "properties": {"name": "Los Angeles"}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-73.9352, 40.7306]
                    },
                    "properties": {"name": "New York"}
                }
            ]
        }"#;

        use oxirs_geosparql::geometry::geojson_parser::parse_geojson_feature_collection;
        let cities = parse_geojson_feature_collection(geojson_collection)?;
        println!("   Parsed {} cities:", cities.len());
        for (i, city) in cities.iter().enumerate() {
            println!("     {}: {}", i + 1, city.to_wkt());
        }

        // 8. Round-trip conversion test
        println!("\n\n=== ROUND-TRIP CONVERSION TEST ===\n");

        let original_geojson = r#"{
            "type": "Point",
            "coordinates": [-122.4194, 37.7749]
        }"#;

        println!("Original GeoJSON:");
        println!("   {}", original_geojson.replace('\n', ""));

        let geometry = Geometry::from_geojson(original_geojson)?;
        println!("\nStep 1 - Parsed to Geometry:");
        println!("   WKT: {}", geometry.to_wkt());

        let converted_geojson = geometry.to_geojson()?;
        println!("\nStep 2 - Converted back to GeoJSON:");
        println!("   {}", converted_geojson.replace('\n', ""));

        let reparsed = Geometry::from_geojson(&converted_geojson)?;
        println!("\nStep 3 - Re-parsed:");
        println!("   WKT: {}", reparsed.to_wkt());

        if geometry.to_wkt() == reparsed.to_wkt() {
            println!("\n✓ Round-trip successful - data preserved!");
        } else {
            println!("\n✗ Round-trip failed - data changed!");
        }

        // 9. Real-world example: Web mapping API response
        println!("\n\n=== REAL-WORLD EXAMPLE: WEB MAPPING API ===\n");

        // Simulating receiving GeoJSON data from a web mapping API
        let building_geojson = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-122.419, 37.774],
                    [-122.418, 37.774],
                    [-122.418, 37.775],
                    [-122.419, 37.775],
                    [-122.419, 37.774]
                ]]
            },
            "properties": {
                "type": "commercial",
                "floors": 25,
                "year_built": 2010
            }
        }"#;

        println!("Received building footprint from web API:");
        let building = Geometry::from_geojson(building_geojson)?;
        println!("   Location: Downtown San Francisco");
        println!("   Geometry Type: {}", building.geometry_type());
        println!("   Spatial Dimension: {}", building.spatial_dimension());

        println!("\nConverting to internal format (WKT):");
        println!("   {}", building.to_wkt());

        println!("\nExporting to GeoJSON for web clients:");
        let export_geojson = building.to_geojson()?;
        println!("   {}", export_geojson.replace('\n', " "));

        // 10. Error handling examples
        println!("\n\n=== ERROR HANDLING ===\n");

        println!("1. Invalid GeoJSON:");
        let invalid_geojson = r#"{"type": "InvalidType", "coordinates": []}"#;
        match Geometry::from_geojson(invalid_geojson) {
            Ok(_) => println!("   Unexpected success"),
            Err(e) => println!("   ✓ Correctly caught error: {}", e),
        }

        println!("\n2. Malformed JSON:");
        let malformed = r#"{"type": "Point", "coordinates": [100.0"#;
        match Geometry::from_geojson(malformed) {
            Ok(_) => println!("   Unexpected success"),
            Err(e) => println!("   ✓ Correctly caught error: {}", e),
        }

        // 11. Integration with WKT
        println!("\n\n=== INTEGRATION WITH WKT ===\n");

        let wkt_geom = Geometry::from_wkt("POINT(139.6917 35.6895)")?;
        println!("WKT Input: POINT(139.6917 35.6895)");
        println!("   Location: Tokyo, Japan");

        println!("\nConverting to GeoJSON:");
        let tokyo_geojson = wkt_geom.to_geojson()?;
        println!("   {}", tokyo_geojson.replace('\n', " "));

        println!("\nConverting back to WKT:");
        let tokyo_reparsed = Geometry::from_geojson(&tokyo_geojson)?;
        println!("   {}", tokyo_reparsed.to_wkt());

        // 12. Interoperability example
        println!("\n\n=== INTEROPERABILITY: GML ↔ GeoJSON ===\n");

        #[cfg(feature = "gml-support")]
        {
            println!("Converting between GML and GeoJSON:");

            let gml_data = r#"<gml:Point xmlns:gml="http://www.opengis.net/gml">
                <gml:pos>-0.1278 51.5074</gml:pos>
            </gml:Point>"#;

            let from_gml = Geometry::from_gml(gml_data)?;
            println!("   GML Input: London coordinates");
            println!("   WKT: {}", from_gml.to_wkt());

            let as_geojson = from_gml.to_geojson()?;
            println!("   GeoJSON Output: {}", as_geojson.replace('\n', " "));

            let back_to_gml = Geometry::from_geojson(&as_geojson)?.to_gml()?;
            println!("   GML Output: {}", back_to_gml.replace('\n', " "));
        }

        #[cfg(not(feature = "gml-support"))]
        {
            println!("(GML support not enabled - enable with 'gml-support' feature)");
        }

        println!("\n=== Example completed successfully! ===");
        println!("\nNote: This example requires the 'geojson-support' feature.");
        println!("Run with: cargo run --example geojson_support --features geojson-support");
    }

    #[cfg(not(feature = "geojson-support"))]
    {
        println!("❌ This example requires the 'geojson-support' feature to be enabled.\n");
        println!("Please run with:");
        println!("   cargo run --example geojson_support --features geojson-support\n");
    }

    Ok(())
}
