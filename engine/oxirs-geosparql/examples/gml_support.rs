//! GML (Geography Markup Language) support example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Parsing GML 3.1.1 and 3.2.1 formats
//! - Converting between GML, WKT, and internal representations
//! - Handling Coordinate Reference Systems (CRS)
//! - Round-trip conversions (GML → Geometry → GML)
//! - Working with various geometry types in GML
//!
//! Run with: cargo run --example gml_support --features gml-support

use oxirs_geosparql::error::Result;

#[cfg(feature = "gml-support")]
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL GML Support Example ===\n");

    #[cfg(feature = "gml-support")]
    {
        // 1. Basic Point parsing
        println!("1. BASIC POINT PARSING:");
        let gml_point = r#"<gml:Point xmlns:gml="http://www.opengis.net/gml">
        <gml:pos>1.0 2.0</gml:pos>
    </gml:Point>"#;

        let point = Geometry::from_gml(gml_point)?;
        println!("   GML Input: {}", gml_point.replace('\n', ""));
        println!("   Parsed WKT: {}", point.to_wkt());
        println!("   Back to GML: {}", point.to_gml()?);

        // 2. Point with CRS
        println!("\n2. POINT WITH COORDINATE REFERENCE SYSTEM:");
        let gml_point_crs = r#"<gml:Point srsName="http://www.opengis.net/def/crs/EPSG/0/4326" xmlns:gml="http://www.opengis.net/gml">
        <gml:pos>-122.4194 37.7749</gml:pos>
    </gml:Point>"#;

        let sf_point = Geometry::from_gml(gml_point_crs)?;
        println!("   Location: San Francisco");
        println!("   GML Input: {}", gml_point_crs.replace('\n', ""));
        println!("   Parsed: {}", sf_point);
        println!("   CRS: {}", sf_point.crs);

        // 3. LineString
        println!("\n3. LINESTRING (PATH/ROUTE):");
        let gml_linestring = r#"<gml:LineString xmlns:gml="http://www.opengis.net/gml">
        <gml:posList>0.0 0.0 10.0 0.0 10.0 10.0 0.0 10.0</gml:posList>
    </gml:LineString>"#;

        let route = Geometry::from_gml(gml_linestring)?;
        println!("   GML Input: {}", gml_linestring.replace('\n', ""));
        println!("   Parsed WKT: {}", route.to_wkt());
        println!("   Geometry Type: {}", route.geometry_type());

        // 4. Polygon (Area)
        println!("\n4. POLYGON (AREA/BOUNDARY):");
        let gml_polygon = r#"<gml:Polygon xmlns:gml="http://www.opengis.net/gml">
        <gml:exterior>
            <gml:LinearRing>
                <gml:posList>0.0 0.0 100.0 0.0 100.0 100.0 0.0 100.0 0.0 0.0</gml:posList>
            </gml:LinearRing>
        </gml:exterior>
    </gml:Polygon>"#;

        let area = Geometry::from_gml(gml_polygon)?;
        println!("   GML Input: {}", gml_polygon.replace('\n', ""));
        println!("   Parsed WKT: {}", area.to_wkt());
        println!("   Spatial Dimension: {}", area.spatial_dimension());

        // 5. Polygon with hole
        println!("\n5. POLYGON WITH HOLE (DONUT SHAPE):");
        let gml_polygon_hole = r#"<gml:Polygon xmlns:gml="http://www.opengis.net/gml">
        <gml:exterior>
            <gml:LinearRing>
                <gml:posList>0.0 0.0 20.0 0.0 20.0 20.0 0.0 20.0 0.0 0.0</gml:posList>
            </gml:LinearRing>
        </gml:exterior>
        <gml:interior>
            <gml:LinearRing>
                <gml:posList>5.0 5.0 15.0 5.0 15.0 15.0 5.0 15.0 5.0 5.0</gml:posList>
            </gml:LinearRing>
        </gml:interior>
    </gml:Polygon>"#;

        let donut = Geometry::from_gml(gml_polygon_hole)?;
        println!("   GML Input: {}", gml_polygon_hole.replace('\n', ""));
        println!("   Parsed WKT: {}", donut.to_wkt());
        println!("   Description: Outer boundary with inner exclusion");

        // 6. MultiPoint
        println!("\n6. MULTIPOINT (COLLECTION OF LOCATIONS):");
        let gml_multipoint = r#"<gml:MultiPoint xmlns:gml="http://www.opengis.net/gml">
        <gml:pointMember>
            <gml:Point><gml:pos>1.0 1.0</gml:pos></gml:Point>
        </gml:pointMember>
        <gml:pointMember>
            <gml:Point><gml:pos>2.0 2.0</gml:pos></gml:Point>
        </gml:pointMember>
        <gml:pointMember>
            <gml:Point><gml:pos>3.0 3.0</gml:pos></gml:Point>
        </gml:pointMember>
    </gml:MultiPoint>"#;

        let locations = Geometry::from_gml(gml_multipoint)?;
        println!("   GML Input: {}", gml_multipoint.replace('\n', ""));
        println!("   Parsed WKT: {}", locations.to_wkt());
        println!("   Use case: Store locations, sensor positions");

        // 7. MultiLineString
        println!("\n7. MULTILINESTRING (MULTIPLE ROUTES):");
        let gml_multilinestring = r#"<gml:MultiLineString xmlns:gml="http://www.opengis.net/gml">
        <gml:lineStringMember>
            <gml:LineString>
                <gml:posList>0.0 0.0 10.0 0.0</gml:posList>
            </gml:LineString>
        </gml:lineStringMember>
        <gml:lineStringMember>
            <gml:LineString>
                <gml:posList>5.0 5.0 15.0 5.0</gml:posList>
            </gml:LineString>
        </gml:lineStringMember>
    </gml:MultiLineString>"#;

        let routes = Geometry::from_gml(gml_multilinestring)?;
        println!("   GML Input: {}", gml_multilinestring.replace('\n', ""));
        println!("   Parsed WKT: {}", routes.to_wkt());
        println!("   Use case: Road networks, shipping lanes");

        // 8. MultiPolygon
        println!("\n8. MULTIPOLYGON (MULTIPLE AREAS):");
        let gml_multipolygon = r#"<gml:MultiPolygon xmlns:gml="http://www.opengis.net/gml">
        <gml:polygonMember>
            <gml:Polygon>
                <gml:exterior>
                    <gml:LinearRing>
                        <gml:posList>0.0 0.0 5.0 0.0 5.0 5.0 0.0 5.0 0.0 0.0</gml:posList>
                    </gml:LinearRing>
                </gml:exterior>
            </gml:Polygon>
        </gml:polygonMember>
        <gml:polygonMember>
            <gml:Polygon>
                <gml:exterior>
                    <gml:LinearRing>
                        <gml:posList>10.0 10.0 15.0 10.0 15.0 15.0 10.0 15.0 10.0 10.0</gml:posList>
                    </gml:LinearRing>
                </gml:exterior>
            </gml:Polygon>
        </gml:polygonMember>
    </gml:MultiPolygon>"#;

        let areas = Geometry::from_gml(gml_multipolygon)?;
        println!("   GML Input: {}", gml_multipolygon.replace('\n', ""));
        println!("   Parsed WKT: {}", areas.to_wkt());
        println!("   Use case: Country territories, land parcels");

        // 9. Round-trip conversion test
        println!("\n\n=== ROUND-TRIP CONVERSION TEST ===\n");

        let original_gml = r#"<gml:Point srsName="http://www.opengis.net/def/crs/EPSG/0/3857" xmlns:gml="http://www.opengis.net/gml">
        <gml:pos>-13648259.0 4546884.0</gml:pos>
    </gml:Point>"#;

        println!("Original GML:");
        println!("   {}", original_gml.replace('\n', ""));

        let geometry = Geometry::from_gml(original_gml)?;
        println!("\nStep 1 - Parsed to Geometry:");
        println!("   WKT: {}", geometry.to_wkt());
        println!("   CRS: {}", geometry.crs);

        let converted_gml = geometry.to_gml()?;
        println!("\nStep 2 - Converted back to GML:");
        println!("   {}", converted_gml.replace('\n', ""));

        let reparsed = Geometry::from_gml(&converted_gml)?;
        println!("\nStep 3 - Re-parsed:");
        println!("   WKT: {}", reparsed.to_wkt());
        println!("   CRS: {}", reparsed.crs);

        if geometry.to_wkt() == reparsed.to_wkt() && geometry.crs == reparsed.crs {
            println!("\n✓ Round-trip successful - data preserved!");
        } else {
            println!("\n✗ Round-trip failed - data changed!");
        }

        // 10. Real-world example: Geographic data exchange
        println!("\n\n=== REAL-WORLD EXAMPLE: GEOGRAPHIC DATA EXCHANGE ===\n");

        // Simulating receiving GML data from an external GIS system
        let building_footprint = r#"<gml:Polygon srsName="http://www.opengis.net/def/crs/EPSG/0/4326" xmlns:gml="http://www.opengis.net/gml">
        <gml:exterior>
            <gml:LinearRing>
                <gml:posList>
                    -122.419 37.774
                    -122.418 37.774
                    -122.418 37.775
                    -122.419 37.775
                    -122.419 37.774
                </gml:posList>
            </gml:LinearRing>
        </gml:exterior>
    </gml:Polygon>"#;

        println!("Received building footprint from GIS system:");
        let building = Geometry::from_gml(building_footprint)?;
        println!("   Location: Downtown San Francisco");
        println!("   CRS: WGS84 (EPSG:4326)");
        println!("   Geometry Type: {}", building.geometry_type());
        println!("   Spatial Dimension: {}", building.spatial_dimension());

        println!("\nConverting to internal format (WKT):");
        println!("   {}", building.to_wkt());

        println!("\nExporting to GML for other systems:");
        let export_gml = building.to_gml()?;
        println!("   {}", export_gml.replace('\n', " "));

        // 11. Error handling examples
        println!("\n\n=== ERROR HANDLING ===\n");

        println!("1. Invalid GML:");
        let invalid_gml = r#"<gml:Point><invalid>test</invalid></gml:Point>"#;
        match Geometry::from_gml(invalid_gml) {
            Ok(_) => println!("   Unexpected success"),
            Err(e) => println!("   ✓ Correctly caught error: {}", e),
        }

        println!("\n2. Unsupported geometry type:");
        let unsupported = r#"<gml:Curve xmlns:gml="http://www.opengis.net/gml">
        <gml:segments><gml:LineStringSegment><gml:posList>0 0 1 1</gml:posList></gml:LineStringSegment></gml:segments>
    </gml:Curve>"#;
        match Geometry::from_gml(unsupported) {
            Ok(_) => println!("   Unexpected success"),
            Err(e) => println!("   ✓ Correctly caught error: {}", e),
        }

        // 12. Integration with WKT
        println!("\n\n=== INTEGRATION WITH WKT ===\n");

        let wkt_geom = Geometry::from_wkt("POINT(139.6917 35.6895)")?;
        println!("WKT Input: POINT(139.6917 35.6895)");
        println!("   Location: Tokyo, Japan");

        println!("\nConverting to GML:");
        let tokyo_gml = wkt_geom.to_gml()?;
        println!("   {}", tokyo_gml.replace('\n', " "));

        println!("\nConverting back to WKT:");
        let tokyo_reparsed = Geometry::from_gml(&tokyo_gml)?;
        println!("   {}", tokyo_reparsed.to_wkt());

        println!("\n=== Example completed successfully! ===");
        println!("\nNote: This example requires the 'gml-support' feature.");
        println!("Run with: cargo run --example gml_support --features gml-support");
    }

    #[cfg(not(feature = "gml-support"))]
    {
        println!("❌ This example requires the 'gml-support' feature to be enabled.\n");
        println!("Please run with:");
        println!("   cargo run --example gml_support --features gml-support\n");
    }

    Ok(())
}
